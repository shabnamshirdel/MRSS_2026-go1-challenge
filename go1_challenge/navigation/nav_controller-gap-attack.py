"""
Navigation Controller base class for Go1 Challenge.

This class handles robot localization and navigation command generation.
@MRSS26: You have to implement the core methods to complete the navigation system.
"""

import numpy as np
import torch
import cv2
import time
import os
from typing import Any

from pyapriltags import Detector

# Position of the tags in the arena
TAG_POSITIONS = {
    0: [1.25, 2.4, 0.5],
    1: [-1.25, 2.4, 0.5],
    2: [-2.4, 1.25, 0.5],
    3: [-2.4, -1.25, 0.5],
    4: [2.4, 1.25, 0.5],
    5: [2.4, -1.25, 0.5],
    6: [1.25, -2.4, 0.5],
    7: [-1.25, -2.4, 0.5],
}  # Tags that are not here are obstacles, not landmarks

# The AprilTags lie on the four arena walls.  The tag frame follows OpenCV's
# convention: +x is image-right, +y is down, and +z points into the wall.
TAG_WALL_NORMALS = {
    0: [0.0, 1.0, 0.0],
    1: [0.0, 1.0, 0.0],
    2: [-1.0, 0.0, 0.0],
    3: [-1.0, 0.0, 0.0],
    4: [1.0, 0.0, 0.0],
    5: [1.0, 0.0, 0.0],
    6: [0.0, -1.0, 0.0],
    7: [0.0, -1.0, 0.0],
}


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class NavController:
    """
    Base Navigation Controller for Go1 robot navigation with AprilTag localization.

    The controller receives observations at regular intervals and maintains internal
    state for robot pose estimation and navigation planning.
    """

    def __init__(
        self, camera_params: tuple[float, float, float, float], tag_size: float = 0.16, tag_family: str = "tag36h11"
    ):
        """
        Initialize the navigation controller.

        Args:
            camera_params: Camera intrinsics (fx, fy, cx, cy) for AprilTag detection
            tag_size: Physical size of AprilTags in meters (length of black square)
            tag_family: AprilTag family name (e.g., "tag36h11")

        Students should implement their state initialization here, including:
        - Robot pose estimates (position, orientation)
        - Map or landmark storage
        - Kalman filter/particle filter initialization
        - Any other navigation-related state
        """
        # Camera and AprilTag parameters
        if isinstance(camera_params, dict):
            camera_params = tuple(camera_params[name] for name in ("fx", "fy", "cx", "cy"))
        if len(camera_params) != 4:
            raise ValueError("camera_params must contain (fx, fy, cx, cy)")
        self.camera_params = tuple(float(value) for value in camera_params)
        self.tag_size = tag_size
        self.tag_family = tag_family

        # AprilTag detector
        self.at_detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Pose/filter parameters.  Unknown-tag exploration uses an arbitrary
        # relative frame; known-tag mode can instead begin highly uncertain.
        self.robot_pose = np.zeros(3, dtype=np.float64)
        self.goal = None  # Current goal position in world frame [x, y]
        self.use_known_tag_localization = False
        # The arbitrary relative-map origin is exact by definition.  Use the
        # large value only when waiting for a known world-frame landmark.
        self.initial_pose_variance = 100.0 if self.use_known_tag_localization else 1.0e-3
        self.pose_variance = self.initial_pose_variance
        self.yaw_variance = self.initial_pose_variance
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.vo_process_variance = 0.04
        self.yaw_process_variance = 0.12
        # Yaw from optical flow is sensitive to blur and vibration during a
        # turn, so let the IMU provide most of the heading increment.
        self.vo_imu_blend = 0.25
        self.minimum_tag_confidence = 15.0

        # The simulated camera is 0.25 m in front of the base and about 0.48 m
        # above the ground.  Only the planar offset enters tag localization.
        self.camera_offset_body = np.array([0.25, 0.0], dtype=np.float64)
        self.camera_height = 0.48

        # Sparse visual-odometry state.
        self.previous_gray = None
        self.previous_depth = None
        self.last_frame_time = None
        self.last_sensor_timestamp = None
        self.camera_frame_dt = 0.1  # Camera observations are produced at 10 Hz in the simulator.
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command = np.zeros(3, dtype=np.float64)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.gravity_filter_gain = 0.20

        # Relative occupancy map.  With unknown landmark positions, the robot's
        # startup pose defines an arbitrary but consistent map frame.  A 12 m
        # square covers the 5 m arena from any random spawn point.
        self.map_resolution = 0.05
        self.map_origin = np.array([-6.0, -6.0], dtype=np.float64)
        self.map_shape = (240, 240)  # rows (y), columns (x)
        self.occupancy_grid = np.zeros(self.map_shape, dtype=np.float32)
        self.visit_grid = np.zeros(self.map_shape, dtype=np.float32)
        self.inflated_grid = np.zeros(self.map_shape, dtype=bool)
        self.log_odds_free = -0.35
        self.log_odds_occupied = 0.85
        self.log_odds_limits = (-5.0, 5.0)
        self.pixel_stride = 12
        self.max_mapping_distance = 5.0
        self.minimum_obstacle_height = 0.10
        self.maximum_obstacle_height = 1.20
        self.enable_occupancy_mapping = True
        self.map_update_count = 0
        self.map_save_interval = 10
        self.map_save_dir = "navigation_maps"
        self._clear_saved_maps()

        # Reactive navigation parameters.  The arena walls are known exactly;
        # the depth camera is used only for temporary obstacles inside it.
        self.robot_inflation_radius = 0.32
        self.obstacle_influence_distance = 0.90
        self.obstacle_detection_range = 1.50
        self.obstacle_repulsion_gain = 1.8
        self.front_blocking_half_width = 0.32
        self.front_blocking_distance = 0.75
        self.front_release_distance = 0.95
        self.emergency_distance = 0.32
        self.maximum_translation_speed = 0.55
        self.maximum_yaw_speed = 0.45
        self.yaw_command_gain = 0.65
        self.navigation_pixel_stride = 10
        self.obstacle_sector_count = 31
        self.local_sector_distances = np.full(
            self.obstacle_sector_count, self.obstacle_detection_range, dtype=np.float64
        )
        self.local_sector_angles = np.linspace(-0.45, 0.45, self.obstacle_sector_count)
        self.best_gap_angle = 0.0
        self.local_obstacle_repulsion = np.zeros(2, dtype=np.float64)
        self.has_depth_observation = False
        self.nearest_front_obstacle = np.inf
        self.left_clearance = self.obstacle_detection_range
        self.right_clearance = self.obstacle_detection_range
        self.avoid_side = 0  # +1 is left, -1 is right.
        self.avoid_clear_frames = 0

        # Exploration state.  Frontiers are free cells adjacent to unknown
        # space; visit counts prevent repeatedly searching the same region.
        self.frontier_update_interval = 5
        self.minimum_frontier_cells = 6
        self.frontier_information_radius = 0.60
        self.selected_frontier = None
        self.frontier_clusters = []
        self.target_detection = None  # Hook for a future dog-bowl detector.
        self.recovery_turn_direction = 1
        self.landmark_loop_closure_gain = 0.08
        self.maximum_landmark_position_correction = 0.04
        self.maximum_landmark_yaw_correction = np.radians(1.5)

        # Tag IDs 0--7 denote wall tags in the supplied arena; other IDs denote
        # obstacle tags.  Their positions are learned in the relative map.
        self.wall_tag_ids = set(TAG_POSITIONS)
        self.landmark_map = {}
        self.detection_buffer = []
        self.has_absolute_pose_fix = False
        self.mapping_frame_initialized = True
        self.state = "EXPLORE"
        self.path = []
        self.last_update_time = None

    def _clear_saved_maps(self) -> None:
        """Remove map snapshots from the previous run once at startup."""
        try:
            os.makedirs(self.map_save_dir, exist_ok=True)
            for filename in os.listdir(self.map_save_dir):
                is_generated_map = filename.startswith("occupancy_map_") and filename.endswith(
                    (".png", ".npy", ".npz")
                )
                if is_generated_map:
                    os.remove(os.path.join(self.map_save_dir, filename))
        except OSError as error:
            print(f"[WARNING] Failed to clear saved occupancy maps: {error}")

    def _roll_pitch_from_gravity(self) -> tuple[float, float]:
        """Estimate body roll and pitch from the filtered gravity direction."""
        gravity = self.projected_gravity
        pitch = np.arcsin(np.clip(gravity[0], -1.0, 1.0))
        roll = np.arctan2(-gravity[1], -gravity[2])
        return float(roll), float(pitch)

    def _body_rotation_world(self) -> np.ndarray:
        """Return the body-to-world rotation using estimated yaw, pitch, and roll."""
        roll, pitch = self._roll_pitch_from_gravity()
        yaw = float(self.robot_pose[2])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rotation_z @ rotation_y @ rotation_x

    def detect_apriltags(self, rgb_image: torch.Tensor, visualize: bool = False) -> dict[int, dict]:
        """
        Detect AprilTags in RGB image and estimate their poses.

        @MRSS26: You can override this method to implement your own detection logic
        or use the raw images for other processing.

        Args:
            rgb_image: RGB image tensor from camera (H, W, 3)
            visualize: Whether to save visualization images to disk

        Returns:
            Dictionary of detected tags with poses in camera frame
            Format: {tag_id: {"pose": {"position": [x,y,z], "rotation_matrix": [[...]]},
                             "distance": float, "confidence": float}}
        """
        try:
            # Convert tensor to numpy and ensure correct format
            if isinstance(rgb_image, torch.Tensor):
                image_np = rgb_image.detach().cpu().numpy()
            else:
                image_np = rgb_image

            # Ensure image is in uint8 format
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            # Convert to grayscale for AprilTag detection
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Detect tags with pose estimation
            tags = self.at_detector.detect(
                gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size
            )

            # --- Process detected tags
            detected_tags = {}

            for tag in tags:
                tag_id = tag.tag_id
                position = tag.pose_t.flatten()  # Translation vector [x, y, z]
                rotation_matrix = tag.pose_R  # 3x3 rotation matrix
                distance = np.linalg.norm(position)

                detected_tags[tag_id] = {
                    "corners": tag.corners.tolist(),
                    "center": tag.center.tolist(),
                    "pose": {
                        "position": position.tolist(),  # [x, y, z] in camera frame
                        "rotation_matrix": rotation_matrix.tolist(),
                    },
                    "distance": float(distance),
                    "confidence": float(tag.decision_margin),
                }

            # Save visualization if requested
            if visualize:
                self._save_visualization_image(
                    image_np, tags, f"tags_{'_'.join(map(str, sorted(detected_tags.keys())))}"
                )

            return detected_tags

        except Exception as e:
            print(f"[ERROR] AprilTag detection failed: {e}")
            return {}

    def _save_visualization_image(self, image_np: np.ndarray, tags: list, prefix: str):
        """Save visualization image with detected tags to disk."""
        try:
            # Convert grayscale to color if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

            # Draw detected tags
            for tag in tags:
                # Draw tag outline
                corners = tag.corners.astype(int)
                for idx in range(len(corners)):
                    cv2.line(vis_image, tuple(corners[idx - 1]), tuple(corners[idx]), (0, 255, 0), 2)

                # Add tag ID text
                center = tag.center.astype(int)
                cv2.putText(
                    vis_image,
                    f"ID:{tag.tag_id}",
                    org=(center[0] - 20, center[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )

                # Add distance text
                distance = np.linalg.norm(tag.pose_t)
                cv2.putText(
                    vis_image,
                    f"{distance:.2f}m",
                    org=(center[0] - 20, center[1] + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                )

            # Save image
            timestamp = int(time.time() * 1000)
            save_dir = "apriltag_detection/"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{save_dir}/{prefix}_{timestamp}.jpg"

            # Convert RGB to BGR for OpenCV saving
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, vis_image_bgr)

            print(f"[INFO] Saved visualization to {filename}")

        except Exception as e:
            print(f"[WARNING] Failed to save visualization: {e}")

    def _estimate_visual_odometry(
        self, previous_gray: np.ndarray, gray: np.ndarray, previous_depth: np.ndarray | None, dt: float
    ) -> tuple[float, float, float, bool]:
        """Estimate planar camera motion with sparse RGB-D (or monocular) VO."""
        features = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=500, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        if features is None or len(features) < 8:
            return 0.0, 0.0, 0.0, False

        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            features,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if tracked is None or status is None:
            return 0.0, 0.0, 0.0, False

        valid = status.reshape(-1).astype(bool)
        previous_pixels = features.reshape(-1, 2)[valid]
        current_pixels = tracked.reshape(-1, 2)[valid]
        if len(previous_pixels) < 8:
            return 0.0, 0.0, 0.0, False

        fx, fy, cx, cy = self.camera_params
        intrinsic_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        rotation_current_from_previous = None
        translation_current_from_previous = None

        # Metric RGB-D VO: back-project features in the previous image and
        # solve their pose in the current image.
        if previous_depth is not None:
            depth = np.asarray(previous_depth).squeeze()
            height, width = depth.shape[:2]
            u = np.rint(previous_pixels[:, 0]).astype(int)
            v = np.rint(previous_pixels[:, 1]).astype(int)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            depths = np.full(len(u), np.nan, dtype=np.float64)
            depths[inside] = depth[v[inside], u[inside]]
            good_depth = np.isfinite(depths) & (depths > 0.10) & (depths < self.max_mapping_distance)

            if np.count_nonzero(good_depth) >= 8:
                z = depths[good_depth]
                pixels_3d = previous_pixels[good_depth]
                object_points = np.column_stack(
                    ((pixels_3d[:, 0] - cx) * z / fx, (pixels_3d[:, 1] - cy) * z / fy, z)
                ).astype(np.float32)
                image_points = current_pixels[good_depth].astype(np.float32)
                try:
                    solved, rvec, tvec, inliers = cv2.solvePnPRansac(
                        object_points,
                        image_points,
                        intrinsic_matrix,
                        None,
                        iterationsCount=100,
                        reprojectionError=2.5,
                        confidence=0.995,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    if solved and inliers is not None and len(inliers) >= 6:
                        rotation_current_from_previous = cv2.Rodrigues(rvec)[0]
                        translation_current_from_previous = tvec.reshape(3)
                except cv2.error:
                    pass

        # Real deployment currently has no depth stream.  Recover the motion
        # direction there and obtain scale from the latest commanded velocity.
        if rotation_current_from_previous is None:
            try:
                essential, mask = cv2.findEssentialMat(
                    previous_pixels,
                    current_pixels,
                    intrinsic_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.5,
                )
                if essential is None:
                    return 0.0, 0.0, 0.0, False
                _, rotation_current_from_previous, unit_translation, _ = cv2.recoverPose(
                    essential, previous_pixels, current_pixels, intrinsic_matrix, mask=mask
                )
                commanded_distance = float(np.linalg.norm(self.latest_velocity_command[:2]) * dt)
                translation_current_from_previous = unit_translation.reshape(3) * commanded_distance
            except cv2.error:
                return 0.0, 0.0, 0.0, False

        # solvePnP/recoverPose returns previous-world points in the current
        # camera frame.  Invert it to get current camera motion in the previous
        # camera frame, then convert OpenCV axes to body x-forward/y-left.
        rotation_previous_from_current = rotation_current_from_previous.T
        camera_delta = -rotation_previous_from_current @ translation_current_from_previous
        forward_axis = rotation_previous_from_current[:, 2]
        delta_yaw = _wrap_angle(np.arctan2(-forward_axis[0], forward_axis[2]))
        delta_forward = float(camera_delta[2])
        delta_left = float(-camera_delta[0])

        # VO measures motion at the camera, not at the robot center.  Because
        # the camera is mounted in front of the base, an in-place turn makes it
        # travel along an arc.  Remove that lever-arm motion before integrating
        # the translation as base motion.
        camera_offset_forward, camera_offset_left = self.camera_offset_body
        cosine_delta = np.cos(delta_yaw)
        sine_delta = np.sin(delta_yaw)
        rotation_translation_forward = (
            cosine_delta * camera_offset_forward
            - sine_delta * camera_offset_left
            - camera_offset_forward
        )
        rotation_translation_left = (
            sine_delta * camera_offset_forward
            + cosine_delta * camera_offset_left
            - camera_offset_left
        )
        delta_forward -= rotation_translation_forward
        delta_left -= rotation_translation_left

        # Reject catastrophic feature matches instead of corrupting the filter.
        maximum_translation = max(0.35, 3.0 * dt)
        if np.hypot(delta_forward, delta_left) > maximum_translation or abs(delta_yaw) > 0.8:
            print(
                "[WARNING] Visual odometry rejected: "
                f"delta_forward={delta_forward:.3f}, delta_left={delta_left:.3f}, "
                f"delta_yaw={np.degrees(delta_yaw):.1f} deg"
            )
            return 0.0, 0.0, 0.0, False
        return delta_forward, delta_left, delta_yaw, True

    def _predict_pose(self, delta_body: np.ndarray, delta_yaw_vo: float, dt: float, vo_valid: bool) -> None:
        """Apply Eq. 1.3--1.6 to the current pose estimate."""
        previous_yaw = float(self.robot_pose[2])
        imu_delta_yaw = self.latest_yaw_rate * dt
        if vo_valid:
            delta_yaw = self.vo_imu_blend * delta_yaw_vo + (1.0 - self.vo_imu_blend) * imu_delta_yaw
        else:
            delta_yaw = imu_delta_yaw

        cosine = np.cos(previous_yaw)
        sine = np.sin(previous_yaw)
        body_to_world = np.array([[cosine, -sine], [sine, cosine]])
        self.robot_pose[:2] += body_to_world @ delta_body
        self.robot_pose[2] = _wrap_angle(previous_yaw + delta_yaw)
        self.pose_variance += self.vo_process_variance * max(dt, 1.0e-3)
        self.yaw_variance += self.yaw_process_variance * max(dt, 1.0e-3)
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])

    @staticmethod
    def _tag_rotation_world(tag_id: int) -> np.ndarray:
        """Return the known tag-to-world rotation for an arena wall tag."""
        tag_z = np.asarray(TAG_WALL_NORMALS[tag_id], dtype=np.float64)
        tag_y = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        tag_x = np.cross(tag_y, tag_z)
        return np.column_stack((tag_x, tag_y, tag_z))

    def _measurement_from_tag(self, tag_id: int, tag_info: dict) -> tuple[np.ndarray, float] | None:
        """Compute a world-frame base pose and variance from one known tag."""
        if tag_id not in TAG_POSITIONS or tag_id not in TAG_WALL_NORMALS:
            return None
        confidence = float(tag_info.get("confidence", 0.0))
        if confidence < self.minimum_tag_confidence:
            return None

        try:
            translation_camera_from_tag = np.asarray(tag_info["pose"]["position"], dtype=np.float64).reshape(3)
            rotation_camera_from_tag = np.asarray(
                tag_info["pose"]["rotation_matrix"], dtype=np.float64
            ).reshape(3, 3)
        except (KeyError, TypeError, ValueError):
            return None

        rotation_world_from_tag = self._tag_rotation_world(tag_id)
        rotation_world_from_camera = rotation_world_from_tag @ rotation_camera_from_tag.T
        camera_forward_world = rotation_world_from_camera[:2, 2]
        if np.linalg.norm(camera_forward_world) < 0.5:
            return None
        yaw = _wrap_angle(np.arctan2(camera_forward_world[1], camera_forward_world[0]))

        tag_world = np.asarray(TAG_POSITIONS[tag_id], dtype=np.float64)
        camera_world = tag_world - rotation_world_from_camera @ translation_camera_from_tag
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        body_to_world = np.array([[cosine, -sine], [sine, cosine]])
        base_xy = camera_world[:2] - body_to_world @ self.camera_offset_body

        distance = float(tag_info.get("distance", np.linalg.norm(translation_camera_from_tag)))
        standard_deviation = 0.025 + 0.02 * distance * distance
        confidence_penalty = max(1.0, 30.0 / max(confidence, 1.0))
        variance = max(1.0e-4, standard_deviation * standard_deviation * confidence_penalty)
        return np.array([base_xy[0], base_xy[1], yaw], dtype=np.float64), variance

    def _fuse_tag_measurements(self, detected_tags: dict[int, dict]) -> None:
        """Fuse tags (Eq. 2.4), then fuse the result with VO (Eq. 2.5--2.6)."""
        measurements = []
        variances = []
        for tag_id, tag_info in detected_tags.items():
            measurement = self._measurement_from_tag(tag_id, tag_info)
            if measurement is not None:
                pose, variance = measurement
                measurements.append(pose)
                variances.append(variance)

        if not measurements:
            return

        poses = np.asarray(measurements)
        weights = 1.0 / np.asarray(variances)
        weight_sum = float(np.sum(weights))
        measured_pose = np.empty(3, dtype=np.float64)
        measured_pose[:2] = np.sum(poses[:, :2] * weights[:, None], axis=0) / weight_sum
        measured_pose[2] = np.arctan2(
            np.sum(weights * np.sin(poses[:, 2])), np.sum(weights * np.cos(poses[:, 2]))
        )
        measured_variance = 1.0 / weight_sum

        # Do not retain scans made in the arbitrary startup frame.  The first
        # confident landmark observation establishes the world-map frame.
        first_absolute_fix = not self.has_absolute_pose_fix

        position_alpha = self.pose_variance / (self.pose_variance + measured_variance)
        yaw_alpha = self.yaw_variance / (self.yaw_variance + measured_variance)
        innovation = measured_pose - self.robot_pose
        innovation[2] = _wrap_angle(innovation[2])
        self.robot_pose[:2] += position_alpha * innovation[:2]
        self.robot_pose[2] += yaw_alpha * innovation[2]
        self.robot_pose[2] = _wrap_angle(self.robot_pose[2])
        self.pose_variance *= 1.0 - position_alpha
        self.yaw_variance *= 1.0 - yaw_alpha
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.has_absolute_pose_fix = True
        if first_absolute_fix:
            self.occupancy_grid.fill(0.0)
            self.map_update_count = 0

        self.detection_buffer.append(
            {
                "tag_ids": sorted(tag_id for tag_id in detected_tags if tag_id in TAG_POSITIONS),
                "pose": measured_pose.tolist(),
            }
        )
        self.detection_buffer = self.detection_buffer[-20:]

    def _world_to_grid(self, point_xy: np.ndarray) -> tuple[int, int] | None:
        cell_xy = np.floor((np.asarray(point_xy) - self.map_origin) / self.map_resolution).astype(int)
        column, row = int(cell_xy[0]), int(cell_xy[1])
        if 0 <= row < self.map_shape[0] and 0 <= column < self.map_shape[1]:
            return row, column
        return None

    def _grid_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """Return the world position at the center of a grid cell."""
        row, column = cell
        return self.map_origin + self.map_resolution * np.array([column + 0.5, row + 0.5])

    def _update_unknown_landmarks(self, detected_tags: dict[int, dict]) -> None:
        """Estimate every observed tag position in the relative exploration map."""
        body_rotation_world = self._body_rotation_world()
        camera_world = np.array(
            [self.robot_pose[0], self.robot_pose[1], self.camera_height], dtype=np.float64
        ) + body_rotation_world @ np.array(
            [self.camera_offset_body[0], self.camera_offset_body[1], 0.0], dtype=np.float64
        )

        for tag_id, tag_info in detected_tags.items():
            confidence = float(tag_info.get("confidence", 0.0))
            if confidence < self.minimum_tag_confidence:
                continue
            try:
                translation_camera = np.asarray(tag_info["pose"]["position"], dtype=np.float64).reshape(3)
                rotation_camera_from_tag = np.asarray(
                    tag_info["pose"]["rotation_matrix"], dtype=np.float64
                ).reshape(3, 3)
            except (KeyError, TypeError, ValueError):
                continue

            # OpenCV camera x/y/z -> body-aligned forward/left/up.
            relative_body = np.array(
                [translation_camera[2], -translation_camera[0], -translation_camera[1]], dtype=np.float64
            )
            position_world = camera_world + body_rotation_world @ relative_body

            tag_normal_camera = rotation_camera_from_tag[:, 2]
            tag_normal_body = np.array(
                [tag_normal_camera[2], -tag_normal_camera[0], -tag_normal_camera[1]], dtype=np.float64
            )
            normal_world = body_rotation_world @ tag_normal_body
            normal_xy_norm = np.linalg.norm(normal_world[:2])
            normal_xy = normal_world[:2] / normal_xy_norm if normal_xy_norm > 1.0e-6 else np.zeros(2)

            distance = float(tag_info.get("distance", np.linalg.norm(translation_camera)))
            measurement_variance = max(1.0e-4, (0.03 + 0.025 * distance * distance) ** 2)
            measurement_weight = confidence / measurement_variance
            kind = "wall" if tag_id in self.wall_tag_ids else "obstacle"
            landmark = self.landmark_map.get(tag_id)
            if landmark is None:
                self.landmark_map[tag_id] = {
                    "position": position_world[:2].tolist(),
                    "normal": normal_xy.tolist(),
                    "variance": measurement_variance,
                    "observations": 1,
                    "weight": measurement_weight,
                    "kind": kind,
                }
                continue

            old_position = np.asarray(landmark["position"], dtype=np.float64)
            residual = position_world[:2] - old_position
            # A bad pose or tag detection must not drag an established landmark
            # across the map.  The gate widens mildly with viewing distance.
            if np.linalg.norm(residual) > 0.35 + 0.08 * distance:
                continue

            if int(landmark["observations"]) >= 5:
                position_correction = -self.landmark_loop_closure_gain * residual
                correction_norm = np.linalg.norm(position_correction)
                if correction_norm > self.maximum_landmark_position_correction:
                    position_correction *= self.maximum_landmark_position_correction / correction_norm
                self.robot_pose[:2] += position_correction

                old_normal = np.asarray(landmark["normal"], dtype=np.float64)
                if np.linalg.norm(old_normal) > 0.5 and np.linalg.norm(normal_xy) > 0.5:
                    yaw_residual = np.arctan2(
                        normal_xy[0] * old_normal[1] - normal_xy[1] * old_normal[0],
                        np.dot(normal_xy, old_normal),
                    )
                    if abs(yaw_residual) < np.radians(20.0):
                        yaw_correction = np.clip(
                            self.landmark_loop_closure_gain * yaw_residual,
                            -self.maximum_landmark_yaw_correction,
                            self.maximum_landmark_yaw_correction,
                        )
                        self.robot_pose[2] = _wrap_angle(self.robot_pose[2] + yaw_correction)

            old_weight = float(landmark["weight"])
            total_weight = old_weight + measurement_weight
            filtered_position = (old_weight * old_position + measurement_weight * position_world[:2]) / total_weight
            old_normal = np.asarray(landmark["normal"], dtype=np.float64)
            filtered_normal = old_weight * old_normal + measurement_weight * normal_xy
            filtered_normal_norm = np.linalg.norm(filtered_normal)
            if filtered_normal_norm > 1.0e-6:
                filtered_normal /= filtered_normal_norm

            landmark.update(
                {
                    "position": filtered_position.tolist(),
                    "normal": filtered_normal.tolist(),
                    "variance": 1.0 / total_weight,
                    "observations": int(landmark["observations"]) + 1,
                    "weight": total_weight,
                }
            )

    def _insert_stable_landmarks(self) -> None:
        """Reinforce mature tag locations as wall/obstacle evidence."""
        for landmark in self.landmark_map.values():
            if int(landmark["observations"]) < 3:
                continue
            position = np.asarray(landmark["position"], dtype=np.float64)
            centers = [position]
            radius_m = 0.22
            if landmark["kind"] == "wall":
                normal = np.asarray(landmark["normal"], dtype=np.float64)
                if np.linalg.norm(normal) > 0.5:
                    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
                    centers = [position + offset * tangent for offset in np.linspace(-0.35, 0.35, 15)]
                radius_m = 0.07
            radius_cells = max(1, int(np.ceil(radius_m / self.map_resolution)))
            for center_position in centers:
                center = self._world_to_grid(center_position)
                if center is None:
                    continue
                center_row, center_column = center
                row_start = max(0, center_row - radius_cells)
                row_end = min(self.map_shape[0], center_row + radius_cells + 1)
                column_start = max(0, center_column - radius_cells)
                column_end = min(self.map_shape[1], center_column + radius_cells + 1)
                for row in range(row_start, row_end):
                    for column in range(column_start, column_end):
                        if (row - center_row) ** 2 + (column - center_column) ** 2 <= radius_cells**2:
                            self.occupancy_grid[row, column] += 0.25

    def _refresh_exploration_layers(self) -> None:
        """Inflate obstacles and select a high-information reachable frontier."""
        occupied = self.occupancy_grid > 0.70
        inflation_cells = max(1, int(np.ceil(self.robot_inflation_radius / self.map_resolution)))
        kernel_size = 2 * inflation_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.inflated_grid = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool)

        free = self.occupancy_grid < -0.45
        unknown = np.abs(self.occupancy_grid) < 0.20
        neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
        adjacent_unknown = cv2.dilate(unknown.astype(np.uint8), neighbor_kernel).astype(bool)
        traversable = free & ~self.inflated_grid
        reachable = traversable
        component_count, traversable_labels = cv2.connectedComponents(traversable.astype(np.uint8), connectivity=8)
        robot_cell = self._world_to_grid(self.robot_pose[:2])
        if component_count > 1 and robot_cell is not None:
            robot_row, robot_column = robot_cell
            reachable_label = int(traversable_labels[robot_row, robot_column])
            if reachable_label == 0:
                search_radius = 6
                row_min = max(0, robot_row - search_radius)
                row_max = min(self.map_shape[0], robot_row + search_radius + 1)
                column_min = max(0, robot_column - search_radius)
                column_max = min(self.map_shape[1], robot_column + search_radius + 1)
                nearby_labels = traversable_labels[row_min:row_max, column_min:column_max]
                nonzero_labels = nearby_labels[nearby_labels > 0]
                if nonzero_labels.size:
                    reachable_label = int(np.bincount(nonzero_labels).argmax())
            if reachable_label > 0:
                reachable = traversable_labels == reachable_label

        frontier_mask = free & adjacent_unknown & reachable
        label_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
            frontier_mask.astype(np.uint8), connectivity=8
        )

        robot_xy = self.robot_pose[:2]
        yaw = float(self.robot_pose[2])
        information_cells = max(1, int(np.ceil(self.frontier_information_radius / self.map_resolution)))
        candidates = []
        for label in range(1, label_count):
            area = int(statistics[label, cv2.CC_STAT_AREA])
            if area < self.minimum_frontier_cells:
                continue
            component_cells = np.argwhere(labels == label)
            centroid_column, centroid_row = centroids[label]
            squared_distance = (
                (component_cells[:, 0] - centroid_row) ** 2 + (component_cells[:, 1] - centroid_column) ** 2
            )
            row, column = component_cells[int(np.argmin(squared_distance))]
            world_position = self._grid_to_world((int(row), int(column)))
            delta = world_position - robot_xy
            travel_distance = float(np.linalg.norm(delta))
            if travel_distance < 0.20:
                continue

            row_min = max(0, int(row) - information_cells)
            row_max = min(self.map_shape[0], int(row) + information_cells + 1)
            column_min = max(0, int(column) - information_cells)
            column_max = min(self.map_shape[1], int(column) + information_cells + 1)
            information_gain = int(np.count_nonzero(unknown[row_min:row_max, column_min:column_max]))
            revisit_cost = float(np.mean(self.visit_grid[row_min:row_max, column_min:column_max]))
            heading_error = abs(_wrap_angle(np.arctan2(delta[1], delta[0]) - yaw))
            score = (
                0.018 * information_gain
                + 0.08 * np.sqrt(area)
                - 0.55 * travel_distance
                - 0.12 * revisit_cost
                - 0.20 * heading_error
            )
            candidates.append(
                {
                    "position": world_position,
                    "score": float(score),
                    "area": area,
                    "information_gain": information_gain,
                }
            )

        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        self.frontier_clusters = candidates[:20]
        if not candidates:
            self.selected_frontier = None
            return

        selected_candidate = candidates[0]
        if self.selected_frontier is not None:
            previous_frontier = np.asarray(self.selected_frontier)
            nearby_candidates = [
                candidate
                for candidate in candidates
                if np.linalg.norm(candidate["position"] - previous_frontier) < 0.35
            ]
            if nearby_candidates and nearby_candidates[0]["score"] >= candidates[0]["score"] - 1.0:
                selected_candidate = nearby_candidates[0]
        self.selected_frontier = selected_candidate["position"].copy()

    @staticmethod
    def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Integer cells along a row/column grid ray, including both ends."""
        row0, column0 = start
        row1, column1 = end
        cells = []
        delta_column = abs(column1 - column0)
        delta_row = -abs(row1 - row0)
        step_column = 1 if column0 < column1 else -1
        step_row = 1 if row0 < row1 else -1
        error = delta_column + delta_row
        while True:
            cells.append((row0, column0))
            if row0 == row1 and column0 == column1:
                break
            twice_error = 2 * error
            if twice_error >= delta_row:
                error += delta_row
                column0 += step_column
            if twice_error <= delta_column:
                error += delta_column
                row0 += step_row
        return cells

    def _update_occupancy_grid(self, distance_image: np.ndarray) -> bool:
        """Back-project depth, clearing floor rays and marking obstacle hits."""
        depth = np.asarray(distance_image).squeeze()
        if depth.ndim != 2:
            return False

        fx, fy, cx, cy = self.camera_params
        body_rotation_world = self._body_rotation_world()
        camera_mount_offset = np.array(
            [self.camera_offset_body[0], self.camera_offset_body[1], 0.0], dtype=np.float64
        )
        camera_world = np.array(
            [self.robot_pose[0], self.robot_pose[1], self.camera_height], dtype=np.float64
        ) + body_rotation_world @ camera_mount_offset
        camera_cell = self._world_to_grid(camera_world[:2])
        if camera_cell is None:
            return False

        visit_radius = max(1, int(np.ceil(0.18 / self.map_resolution)))
        camera_row, camera_column = camera_cell
        for row in range(max(0, camera_row - visit_radius), min(self.map_shape[0], camera_row + visit_radius + 1)):
            for column in range(
                max(0, camera_column - visit_radius), min(self.map_shape[1], camera_column + visit_radius + 1)
            ):
                if (row - camera_row) ** 2 + (column - camera_column) ** 2 <= visit_radius**2:
                    self.visit_grid[row, column] = min(1000.0, self.visit_grid[row, column] + 1.0)
        valid_ray_count = 0

        for v in range(0, depth.shape[0], self.pixel_stride):
            for u in range(0, depth.shape[1], self.pixel_stride):
                distance = float(depth[v, u])
                if not np.isfinite(distance) or distance <= 0.10 or distance > self.max_mapping_distance:
                    continue
                camera_x = (u - cx) * distance / fx
                camera_y = (v - cy) * distance / fy
                # OpenCV camera axes are x-right, y-down, z-forward.  The
                # aligned body axes are x-forward, y-left, z-up.
                point_camera_in_body_axes = np.array([distance, -camera_x, -camera_y], dtype=np.float64)
                point_world = camera_world + body_rotation_world @ point_camera_in_body_axes
                hit_cell = self._world_to_grid(point_world[:2])
                if hit_cell is None:
                    continue

                ray = self._bresenham(camera_cell, hit_cell)
                endpoint_is_obstacle = self.minimum_obstacle_height <= point_world[2] <= self.maximum_obstacle_height
                free_cells = ray[:-1] if endpoint_is_obstacle else ray
                for row, column in free_cells:
                    self.occupancy_grid[row, column] += self.log_odds_free
                if endpoint_is_obstacle:
                    hit_row, hit_column = ray[-1]
                    self.occupancy_grid[hit_row, hit_column] += self.log_odds_occupied
                valid_ray_count += 1

        self._insert_stable_landmarks()
        np.clip(self.occupancy_grid, *self.log_odds_limits, out=self.occupancy_grid)
        if valid_ray_count == 0:
            return False

        self.map_update_count += 1
        if self.map_update_count % self.frontier_update_interval == 0:
            self._refresh_exploration_layers()
        if self.map_update_count % self.map_save_interval == 0:
            self._save_occupancy_map()
        return True

    def _save_occupancy_map(self) -> None:
        """Persist both numeric log odds and a viewable PNG every ten updates."""
        try:
            os.makedirs(self.map_save_dir, exist_ok=True)
            stem = os.path.join(self.map_save_dir, f"occupancy_map_{self.map_update_count:06d}")
            np.save(f"{stem}.npy", self.occupancy_grid)
            np.savez_compressed(
                f"{stem}_layers.npz",
                occupancy=self.occupancy_grid,
                visits=self.visit_grid,
                inflated=self.inflated_grid,
            )
            probability = 1.0 / (1.0 + np.exp(-self.occupancy_grid))
            image = np.flipud(np.rint(255.0 * (1.0 - probability)).astype(np.uint8))
            if not cv2.imwrite(f"{stem}.png", image):
                raise OSError("cv2.imwrite returned false")

            exploration_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            inflated_image_mask = np.flipud(self.inflated_grid & (self.occupancy_grid <= 0.70))
            exploration_image[inflated_image_mask] = (0, 165, 255)
            for candidate in self.frontier_clusters:
                cell = self._world_to_grid(candidate["position"])
                if cell is not None:
                    row, column = cell
                    cv2.circle(exploration_image, (column, self.map_shape[0] - 1 - row), 2, (255, 0, 0), -1)
            robot_cell = self._world_to_grid(self.robot_pose[:2])
            if robot_cell is not None:
                row, column = robot_cell
                cv2.circle(exploration_image, (column, self.map_shape[0] - 1 - row), 3, (0, 255, 0), -1)
            if self.selected_frontier is not None:
                frontier_cell = self._world_to_grid(self.selected_frontier)
                if frontier_cell is not None:
                    row, column = frontier_cell
                    cv2.circle(exploration_image, (column, self.map_shape[0] - 1 - row), 4, (0, 0, 255), 1)
            if not cv2.imwrite(f"{stem}_exploration.png", exploration_image):
                raise OSError("failed to save exploration visualization")
            print(f"[NavController] Saved occupancy and exploration maps at {stem}")
        except (OSError, ValueError, cv2.error) as error:
            print(f"[WARNING] Failed to save occupancy map: {error}")

    def _update_local_obstacles(self, distance_image: np.ndarray) -> None:
        """Build a bounded local repulsive force from one nearest hit per sector."""
        depth = np.asarray(distance_image).squeeze()
        self.local_obstacle_repulsion.fill(0.0)
        self.nearest_front_obstacle = np.inf
        self.left_clearance = self.obstacle_detection_range
        self.right_clearance = self.obstacle_detection_range
        if depth.ndim != 2:
            return
        self.has_depth_observation = True

        fx, fy, cx, cy = self.camera_params
        body_rotation_world = self._body_rotation_world()
        camera_mount_offset = np.array(
            [self.camera_offset_body[0], self.camera_offset_body[1], 0.0], dtype=np.float64
        )
        camera_world = np.array(
            [self.robot_pose[0], self.robot_pose[1], self.camera_height], dtype=np.float64
        ) + body_rotation_world @ camera_mount_offset
        cosine = np.cos(self.robot_pose[2])
        sine = np.sin(self.robot_pose[2])
        world_to_planar_body = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)

        sector_distances = np.full(self.obstacle_sector_count, np.inf, dtype=np.float64)
        sector_points = np.zeros((self.obstacle_sector_count, 2), dtype=np.float64)
        half_fov = np.arctan2(depth.shape[1] * 0.5, fx)
        self.local_sector_angles = np.linspace(-half_fov, half_fov, self.obstacle_sector_count)

        for v in range(0, depth.shape[0], self.navigation_pixel_stride):
            for u in range(0, depth.shape[1], self.navigation_pixel_stride):
                distance = float(depth[v, u])
                if not np.isfinite(distance) or not 0.10 < distance < self.obstacle_detection_range:
                    continue

                camera_x = (u - cx) * distance / fx
                camera_y = (v - cy) * distance / fy
                point_camera_in_body_axes = np.array([distance, -camera_x, -camera_y], dtype=np.float64)
                point_world = camera_world + body_rotation_world @ point_camera_in_body_axes
                if point_world[2] < self.minimum_obstacle_height or point_world[2] > 1.20:
                    continue

                point_body = world_to_planar_body @ (point_world[:2] - self.robot_pose[:2])
                planar_distance = float(np.linalg.norm(point_body))
                if planar_distance <= 0.10 or point_body[0] < -0.05:
                    continue

                bearing = float(np.arctan2(point_body[1], point_body[0]))
                normalized_bearing = np.clip((bearing + half_fov) / (2.0 * half_fov), 0.0, 1.0)
                sector = min(int(normalized_bearing * self.obstacle_sector_count), self.obstacle_sector_count - 1)
                if planar_distance < sector_distances[sector]:
                    sector_distances[sector] = planar_distance
                    sector_points[sector] = point_body

        repulsion = np.zeros(2, dtype=np.float64)
        active_sector_count = 0
        left_samples = []
        right_samples = []
        side_split_angle = 0.08
        for distance, point_body in zip(sector_distances, sector_points):
            if not np.isfinite(distance):
                continue
            bearing = float(np.arctan2(point_body[1], point_body[0]))
            if bearing >= side_split_angle:
                left_samples.append(distance)
            elif bearing <= -side_split_angle:
                right_samples.append(distance)

            if point_body[0] > 0.0 and abs(point_body[1]) < self.front_blocking_half_width:
                self.nearest_front_obstacle = min(self.nearest_front_obstacle, distance)

            if distance < self.obstacle_influence_distance:
                strength = (self.obstacle_influence_distance - distance) / self.obstacle_influence_distance
                repulsion -= strength * point_body / distance
                active_sector_count += 1

        if active_sector_count:
            repulsion *= self.obstacle_repulsion_gain / active_sector_count
        self.local_obstacle_repulsion = repulsion
        self.local_sector_distances = np.where(
            np.isfinite(sector_distances), sector_distances, self.obstacle_detection_range
        )
        if left_samples:
            self.left_clearance = float(np.mean(left_samples))
        if right_samples:
            self.right_clearance = float(np.mean(right_samples))

        front_blocked = self.nearest_front_obstacle < self.front_blocking_distance
        if front_blocked:
            self.avoid_clear_frames = 0
            if self.avoid_side == 0:
                self.avoid_side = 1 if self.left_clearance >= self.right_clearance else -1
            self.state = "AVOID_LEFT" if self.avoid_side > 0 else "AVOID_RIGHT"
        elif self.avoid_side:
            if self.nearest_front_obstacle > self.front_release_distance:
                self.avoid_clear_frames += 1
            else:
                self.avoid_clear_frames = 0
            if self.avoid_clear_frames >= 4:
                self.avoid_side = 0
                self.avoid_clear_frames = 0
                self.state = "EXPLORE_GAP"

    def _direction_exploration_score(self, angle_body: float) -> float:
        """Reward unknown, unvisited map cells along a candidate local direction."""
        direction_world = np.array(
            [np.cos(self.robot_pose[2] + angle_body), np.sin(self.robot_pose[2] + angle_body)]
        )
        score = 0.0
        for distance in (0.35, 0.60, 0.90, 1.20):
            cell = self._world_to_grid(self.robot_pose[:2] + distance * direction_world)
            if cell is None:
                score -= 3.0
                continue
            row, column = cell
            if self.inflated_grid[row, column]:
                score -= 4.0
            elif abs(float(self.occupancy_grid[row, column])) < 0.20:
                score += 1.3
            elif self.occupancy_grid[row, column] < -0.45:
                score += 0.25
            score -= 0.06 * min(float(self.visit_grid[row, column]), 20.0)
        return score

    def _select_gap_direction(self, preferred_angle: float | None) -> float:
        """Choose a wide visible gap, biased toward a frontier and unseen cells."""
        distances = self.local_sector_distances
        angles = self.local_sector_angles
        if distances.size == 0:
            return 0.0

        blocked = np.zeros(distances.size, dtype=bool)
        angular_step = max(1.0e-3, float(abs(angles[1] - angles[0]))) if len(angles) > 1 else 0.05
        for index, distance in enumerate(distances):
            if distance >= self.obstacle_detection_range:
                continue
            inflation_angle = np.arctan2(self.robot_inflation_radius, max(distance, 0.10))
            inflation_sectors = int(np.ceil(inflation_angle / angular_step))
            start = max(0, index - inflation_sectors)
            end = min(len(blocked), index + inflation_sectors + 1)
            if distance < self.front_release_distance:
                blocked[start:end] = True

        gaps = []
        start = None
        for index, is_blocked in enumerate(np.append(blocked, True)):
            if not is_blocked and start is None:
                start = index
            elif is_blocked and start is not None:
                if index - start >= 2:
                    gaps.append((start, index - 1))
                start = None

        if not gaps:
            self.state = "RECOVERY"
            self.recovery_turn_direction = self.avoid_side or self.recovery_turn_direction
            self.best_gap_angle = float(self.recovery_turn_direction * max(abs(angles[0]), abs(angles[-1])))
            return self.best_gap_angle

        if self.state == "RECOVERY":
            self.state = "EXPLORE_GAP"

        best_score = -np.inf
        best_angle = 0.0
        for gap_start, gap_end in gaps:
            candidate_indices = range(gap_start, gap_end + 1)
            for index in candidate_indices:
                angle = float(angles[index])
                width = float(angles[gap_end] - angles[gap_start] + angular_step)
                clearance = float(np.mean(distances[gap_start : gap_end + 1]))
                frontier_alignment = 0.0 if preferred_angle is None else abs(_wrap_angle(angle - preferred_angle))
                score = (
                    1.8 * width
                    + 0.65 * clearance
                    + self._direction_exploration_score(angle)
                    - 0.85 * frontier_alignment
                    - 0.25 * abs(angle)
                )
                if score > best_score:
                    best_score = score
                    best_angle = angle

        self.best_gap_angle = best_angle
        return best_angle

    def update(self, observations: dict[str, Any]) -> None:
        """
        Update internal navigation state based on sensor observations.

        This method is called from the main loop (sim or real) when new information is available.

        @MRSS26: It's here where you should implement localization and planning logic here.

        For the sim environemnt, frame HxW is 480x640

        Args:
            observations: Dictionary containing:
                - 'base_ang_vel': Angular velocity [rad/s] (3,) array [roll_rate, pitch_rate, yaw_rate]
                - 'projected_gravity': Gravity vector in body frame (3,) array
                - 'velocity_commands': Current velocity commands (3,) array [vx, vy, wz]
                - 'joint_pos': Joint positions (12,) array
                - 'joint_vel': Joint velocities (12,) array
                - 'actions': Last action taken (12,) array
                - 'goal_position': Goal position in world frame (2,) array [x, y]
                - 'robot_pose': Robot position in world frame (3,) array [x, y, yaw]. This is ONLY to help you debug
                - 'camera_rgb': RGB camera image tensor (H, W, 3)
                - 'camera_distance': Distance to camera in meters (H, W, 1)
                and validate your localization logic. It will not be available on the real robot.

        Note:
            - Not all observations are guaranteed to be present. This can be called with only robot observations
            or only camera observations.
            - rgb_image may be None if camera is not available
        """
        # Proprioception and camera data arrive in separate calls.  Cache the
        # latest IMU/command values without integrating twice.
        
        print("[NavController] observations keys:", list(observations.keys()))
        # print("[NavController] observations:", observations)

        base_ang_vel = observations.get("base_ang_vel")
        if base_ang_vel is not None:
            angular_velocity = np.asarray(base_ang_vel, dtype=np.float64).reshape(-1)
            if angular_velocity.size >= 3:
                self.latest_yaw_rate = float(angular_velocity[2])

        projected_gravity = observations.get("projected_gravity")
        if projected_gravity is not None:
            measured_gravity = np.asarray(projected_gravity, dtype=np.float64).reshape(-1)
            if measured_gravity.size >= 3 and np.all(np.isfinite(measured_gravity[:3])):
                measured_gravity = measured_gravity[:3]
                gravity_norm = np.linalg.norm(measured_gravity)
                if gravity_norm > 1.0e-6:
                    measured_gravity /= gravity_norm
                    gain = self.gravity_filter_gain
                    self.projected_gravity = (1.0 - gain) * self.projected_gravity + gain * measured_gravity
                    self.projected_gravity /= np.linalg.norm(self.projected_gravity)

        velocity_command = observations.get("velocity_commands")
        if velocity_command is not None:
            command = np.asarray(velocity_command, dtype=np.float64).reshape(-1)
            if command.size >= 3:
                self.latest_velocity_command = command[:3].copy()

        goal_position = observations.get("goal_position")
        if goal_position is not None:
            self.goal = np.asarray(goal_position, dtype=np.float64).reshape(-1)[:2]

        rgb_image = observations.get("camera_rgb")
        distance_image = observations.get("camera_distance")
        if rgb_image is None:
            return

        if isinstance(rgb_image, torch.Tensor):
            image = rgb_image.detach().cpu().numpy()
        else:
            image = np.asarray(rgb_image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.size and image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        depth = None if distance_image is None else np.asarray(distance_image).squeeze()

        wall_timestamp = time.monotonic()
        sensor_timestamp = observations.get("timestamp")
        if self.previous_gray is None:
            dt = 0.0
        elif sensor_timestamp is not None and self.last_sensor_timestamp is not None:
            dt = float(np.clip(float(sensor_timestamp) - self.last_sensor_timestamp, 1.0e-3, 0.5))
        else:
            # Isaac Sim may execute faster or slower than real time.  Therefore,
            # wall-clock time cannot be used to integrate simulated IMU rates.
            dt = self.camera_frame_dt

        delta_body = np.zeros(2, dtype=np.float64)
        delta_yaw_vo = 0.0
        vo_valid = False
        if self.previous_gray is not None and dt > 0.0:
            delta_forward, delta_left, delta_yaw_vo, vo_valid = self._estimate_visual_odometry(
                self.previous_gray, gray, self.previous_depth, dt
            )
            delta_body[:] = (delta_forward, delta_left)
            self._predict_pose(delta_body, delta_yaw_vo, dt, vo_valid)

        detected_tags = self.detect_apriltags(image, visualize=False)
        if self.use_known_tag_localization:
            self._fuse_tag_measurements(detected_tags)
        self._update_unknown_landmarks(detected_tags)

        if depth is not None:
            self._update_local_obstacles(depth)

        # Camera frames already arrive at 10 Hz.  Update once per frame rather
        # than using wall time, because Isaac Sim may run faster than real time.
        should_update_map = (
            self.enable_occupancy_mapping and depth is not None and self.mapping_frame_initialized
        )
        if should_update_map:
            self._update_occupancy_grid(depth)

        if not self.path or np.linalg.norm(self.robot_pose[:2] - np.asarray(self.path[-1])) >= 0.10:
            self.path.append(self.robot_pose[:2].copy().tolist())
            self.path = self.path[-2000:]

        self.previous_gray = gray.copy()
        self.previous_depth = None if depth is None else depth.copy()
        self.last_frame_time = wall_timestamp
        self.last_sensor_timestamp = None if sensor_timestamp is None else float(sensor_timestamp)
        self.last_update_time = wall_timestamp

        print(
            "[NavController] Estimated pose: "
            f"x={self.robot_pose[0]:.3f} m, y={self.robot_pose[1]:.3f} m, "
            f"yaw={np.degrees(self.robot_pose[2]):.1f} deg, "
            f"xy_var={self.pose_variance:.5f}, yaw_var={self.yaw_variance:.5f}"
        )


    def get_command(self) -> np.ndarray:
        """
        Generate an exploration command from frontiers and visible free gaps.

        Returns:
            Array (3, ) with the structure  [lin_vel_x, lin_vel_y, ang_vel_z] in robot body frame.
            All values should be in range [-1, 1] representing normalized velocities.
        """
        # self.goal is intentionally ignored during search.  target_detection
        # is reserved for the future bowl detector and approach controller.
        if not self.has_depth_observation:
            self.state = "WAIT_FOR_DEPTH"
            return np.zeros(3, dtype=np.float32)

        preferred_angle = None
        if self.selected_frontier is not None:
            frontier_delta = np.asarray(self.selected_frontier) - self.robot_pose[:2]
            frontier_distance = float(np.linalg.norm(frontier_delta))
            if frontier_distance < 0.25:
                self.selected_frontier = None
            else:
                frontier_heading_world = np.arctan2(frontier_delta[1], frontier_delta[0])
                preferred_angle = _wrap_angle(frontier_heading_world - self.robot_pose[2])

        gap_angle = self._select_gap_direction(preferred_angle)
        if self.state == "RECOVERY":
            return np.array(
                [0.0, 0.0, np.clip(0.8 * gap_angle, -self.maximum_yaw_speed, self.maximum_yaw_speed)],
                dtype=np.float32,
            )

        self.state = "EXPLORE_FRONTIER" if self.selected_frontier is not None else "EXPLORE_GAP"
        desired_translation = np.array([np.cos(gap_angle), 0.45 * np.sin(gap_angle)], dtype=np.float64)
        desired_translation += 0.35 * self.local_obstacle_repulsion

        # The lateral term is deliberately smaller than forward motion because
        # the camera cannot observe the robot's sides.
        if self.avoid_side:
            desired_translation += np.array([0.10, 0.35 * self.avoid_side], dtype=np.float64)
            self.state = "AVOID_LEFT" if self.avoid_side > 0 else "AVOID_RIGHT"

        desired_norm = float(np.linalg.norm(desired_translation))
        if desired_norm < 1.0e-6:
            desired_translation[:] = (1.0, 0.0)
            desired_norm = 1.0
        desired_direction = desired_translation / desired_norm

        front_clearance_factor = np.clip(
            (self.nearest_front_obstacle - self.emergency_distance)
            / max(self.front_blocking_distance - self.emergency_distance, 1.0e-3),
            0.0,
            1.0,
        )
        if not np.isfinite(self.nearest_front_obstacle):
            front_clearance_factor = 1.0
        translation_speed = self.maximum_translation_speed * (0.30 + 0.70 * front_clearance_factor)
        lin_vel_x = translation_speed * max(0.0, desired_direction[0])
        lin_vel_y = translation_speed * desired_direction[1]
        ang_vel_z = np.clip(
            self.yaw_command_gain * gap_angle,
            -self.maximum_yaw_speed,
            self.maximum_yaw_speed,
        )

        if self.nearest_front_obstacle < self.emergency_distance:
            lin_vel_x = 0.0
            emergency_side = self.avoid_side or self.recovery_turn_direction
            lin_vel_y = 0.30 * emergency_side
            ang_vel_z = self.maximum_yaw_speed * emergency_side
            self.state = "EMERGENCY_AVOID"

        command = np.array(
            [
                np.clip(lin_vel_x, -1.0, 1.0),
                np.clip(lin_vel_y, -1.0, 1.0),
                np.clip(ang_vel_z, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return command

    def reset(self) -> None:
        """
        Reset the navigation controller state.

        Called when the environment/robot is reset.
        """
        self.robot_pose = np.zeros(3, dtype=np.float64)
        self.goal = None
        self.pose_variance = self.initial_pose_variance
        self.yaw_variance = self.initial_pose_variance
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.previous_gray = None
        self.previous_depth = None
        self.last_frame_time = None
        self.last_sensor_timestamp = None
        self.last_update_time = None
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command.fill(0.0)
        self.projected_gravity[:] = (0.0, 0.0, -1.0)
        self.occupancy_grid.fill(0.0)
        self.visit_grid.fill(0.0)
        self.inflated_grid.fill(False)
        self.map_update_count = 0
        self.local_obstacle_repulsion.fill(0.0)
        self.has_depth_observation = False
        self.local_sector_distances.fill(self.obstacle_detection_range)
        self.best_gap_angle = 0.0
        self.nearest_front_obstacle = np.inf
        self.left_clearance = self.obstacle_detection_range
        self.right_clearance = self.obstacle_detection_range
        self.avoid_side = 0
        self.avoid_clear_frames = 0
        self.landmark_map.clear()
        self.detection_buffer.clear()
        self.has_absolute_pose_fix = False
        self.mapping_frame_initialized = True
        self.selected_frontier = None
        self.frontier_clusters.clear()
        self.target_detection = None
        self.recovery_turn_direction = 1
        self.state = "EXPLORE"
        self.path.clear()

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information for visualization/logging."""

        return {
            "robot_pose": self.robot_pose.tolist(),
            "pose_covariance": self.pose_covariance.tolist(),
            "pose_variance": self.pose_variance,
            "yaw_variance": self.yaw_variance,
            "projected_gravity": self.projected_gravity.tolist(),
            "has_absolute_pose_fix": self.has_absolute_pose_fix,
            "mapping_frame_initialized": self.mapping_frame_initialized,
            "landmark_count": len(self.landmark_map),
            "landmark_map": self.landmark_map,
            "state": self.state,
            "goal": None if self.goal is None else self.goal.tolist(),
            "local_obstacle_repulsion": self.local_obstacle_repulsion.tolist(),
            "nearest_front_obstacle": self.nearest_front_obstacle,
            "left_clearance": self.left_clearance,
            "right_clearance": self.right_clearance,
            "avoid_side": self.avoid_side,
            "best_gap_angle": self.best_gap_angle,
            "selected_frontier": (
                None if self.selected_frontier is None else np.asarray(self.selected_frontier).tolist()
            ),
            "frontier_count": len(self.frontier_clusters),
            "path": self.path,
            "map_update_count": self.map_update_count,
            "map_resolution": self.map_resolution,
            "map_origin": self.map_origin.tolist(),
            "occupancy_grid": self.occupancy_grid,
            "visit_grid": self.visit_grid,
            "inflated_grid": self.inflated_grid,
            "camera_params": self.camera_params,
            "tag_size": self.tag_size,
            "tag_family": self.tag_family,
        }
