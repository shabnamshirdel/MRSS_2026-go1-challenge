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

        # Pose/filter parameters.  Variance is deliberately large until the
        # first confident landmark fix, as requested by Eq. 1.6 and 2.5.
        self.robot_pose = np.zeros(3, dtype=np.float64)
        self.goal = None  # Current goal position in world frame [x, y]
        self.initial_pose_variance = 100.0
        self.pose_variance = self.initial_pose_variance
        self.yaw_variance = self.initial_pose_variance
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.vo_process_variance = 0.04
        self.yaw_process_variance = 0.12
        self.vo_imu_blend = 0.65
        self.minimum_tag_confidence = 15.0

        # The simulated camera is 0.25 m in front of the base and about 0.48 m
        # above the ground.  Only the planar offset enters tag localization.
        self.camera_offset_body = np.array([0.25, 0.0], dtype=np.float64)
        self.camera_height = 0.48

        # Sparse visual-odometry state.
        self.previous_gray = None
        self.previous_depth = None
        self.last_frame_time = None
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command = np.zeros(3, dtype=np.float64)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.gravity_filter_gain = 0.20

        # Occupancy map: x/y in [-3, 3] m at 5 cm/cell.  Log odds of zero
        # represents unknown space.
        self.map_resolution = 0.05
        self.map_origin = np.array([-3.0, -3.0], dtype=np.float64)
        self.map_shape = (120, 120)  # rows (y), columns (x)
        self.occupancy_grid = np.zeros(self.map_shape, dtype=np.float32)
        self.log_odds_free = -0.35
        self.log_odds_occupied = 0.85
        self.log_odds_limits = (-5.0, 5.0)
        self.pixel_stride = 12
        self.max_mapping_distance = 5.0
        self.minimum_obstacle_height = 0.10
        self.map_update_count = 0
        self.map_update_interval = 0.1
        self.last_map_update_time = None
        self.map_save_interval = 10
        self.map_save_dir = "navigation_maps"

        self.landmark_map = {tag_id: list(position) for tag_id, position in TAG_POSITIONS.items()}
        self.detection_buffer = []
        self.state = "EXPLORE"
        self.path = []
        self.last_update_time = None

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

        # Reject catastrophic feature matches instead of corrupting the filter.
        maximum_translation = max(0.35, 3.0 * dt)
        if np.hypot(delta_forward, delta_left) > maximum_translation or abs(delta_yaw) > 0.8:
            print("[WARNING] Visual odometry rejected: "
                f"delta_forward={delta_forward:.3f}, delta_left={delta_left:.3f}, delta_yaw={np.degrees(delta_yaw):.1f} deg")
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
        """Back-project sampled depth pixels and apply inverse sensor log odds."""
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
                if point_world[2] < self.minimum_obstacle_height or point_world[2] > 1.5:
                    continue

                hit_cell = self._world_to_grid(point_world[:2])
                if hit_cell is None:
                    continue

                ray = self._bresenham(camera_cell, hit_cell)
                for row, column in ray[:-1]:
                    self.occupancy_grid[row, column] += self.log_odds_free
                hit_row, hit_column = ray[-1]
                self.occupancy_grid[hit_row, hit_column] += self.log_odds_occupied
                valid_ray_count += 1

        np.clip(self.occupancy_grid, *self.log_odds_limits, out=self.occupancy_grid)
        if valid_ray_count == 0:
            return False

        self.map_update_count += 1
        if self.map_update_count % self.map_save_interval == 0:
            self._save_occupancy_map()
        return True

    def _save_occupancy_map(self) -> None:
        """Persist both numeric log odds and a viewable PNG every ten updates."""
        try:
            os.makedirs(self.map_save_dir, exist_ok=True)
            stem = os.path.join(self.map_save_dir, f"occupancy_map_{self.map_update_count:06d}")
            np.save(f"{stem}.npy", self.occupancy_grid)
            probability = 1.0 / (1.0 + np.exp(-self.occupancy_grid))
            image = np.flipud(np.rint(255.0 * (1.0 - probability)).astype(np.uint8))
            if not cv2.imwrite(f"{stem}.png", image):
                raise OSError("cv2.imwrite returned false")
            print(f"[NavController] Saved occupancy map to {stem}.png/.npy")
        except (OSError, ValueError, cv2.error) as error:
            print(f"[WARNING] Failed to save occupancy map: {error}")

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

        timestamp = float(observations.get("timestamp", time.monotonic()))
        if self.last_frame_time is None:
            dt = 0.0
        else:
            dt = float(np.clip(timestamp - self.last_frame_time, 1.0e-3, 0.5))

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
        self._fuse_tag_measurements(detected_tags)

        map_update_due = (
            self.last_map_update_time is None
            or timestamp - self.last_map_update_time >= self.map_update_interval
            or timestamp < self.last_map_update_time
        )
        if depth is not None and map_update_due and self._update_occupancy_grid(depth):
            self.last_map_update_time = timestamp

        self.previous_gray = gray.copy()
        self.previous_depth = None if depth is None else depth.copy()
        self.last_frame_time = timestamp
        self.last_update_time = timestamp

        print(
            "[NavController] Estimated pose: "
            f"x={self.robot_pose[0]:.3f} m, y={self.robot_pose[1]:.3f} m, "
            f"yaw={np.degrees(self.robot_pose[2]):.1f} deg, "
            f"xy_var={self.pose_variance:.5f}, yaw_var={self.yaw_variance:.5f}"
        )

        real_pose = observations.get('robot_pose', None)
        if real_pose is not None:
            print(
                "[NavController] Ground truth pose: "
                f"x={real_pose[0]:.3f} m, y={real_pose[1]:.3f} m, "
                f"yaw={np.degrees(real_pose[2]):.1f} deg"
            )
        else:
            print("[NavController] Ground truth pose: Not available")

    def get_command(self) -> np.ndarray:
        """
        Generate velocity command based on current navigation state.

        @MRSS26: You should implement navigation/path planning logic here. Ideas of observations you could use:
        - Current robot pose estimate (self.robot_pose)
        - Goal position (from latest observations)
        - Detected AprilTags for localization
        - Obstacle avoidance logic

        Returns:
            Array (3, ) with the structure  [lin_vel_x, lin_vel_y, ang_vel_z] in robot body frame.
            All values should be in range [-1, 1] representing normalized velocities.
        """
        # Example placeholder - simple forward motion:
        lin_vel_x = 0.0  # Move forward at reduced speed
        lin_vel_y = 0.0  # No lateral motion
        ang_vel_z = 0.0  # No rotation

        # TODO MRSS26: Implement navigation logic

        # Ensure commands are in valid range
        lin_vel_x = np.clip(lin_vel_x, -1.0, 1.0)
        lin_vel_y = np.clip(lin_vel_y, -1.0, 1.0)
        ang_vel_z = np.clip(ang_vel_z, -1.0, 1.0)

        command = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)

        return command

    def reset(self) -> None:
        """
        Reset the navigation controller state.

        Called when the environment/robot is reset.
        """
        self.robot_pose = np.zeros(3, dtype=np.float64)
        self.pose_variance = self.initial_pose_variance
        self.yaw_variance = self.initial_pose_variance
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.previous_gray = None
        self.previous_depth = None
        self.last_frame_time = None
        self.last_update_time = None
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command.fill(0.0)
        self.projected_gravity[:] = (0.0, 0.0, -1.0)
        self.occupancy_grid.fill(0.0)
        self.map_update_count = 0
        self.last_map_update_time = None
        self.landmark_map = {tag_id: list(position) for tag_id, position in TAG_POSITIONS.items()}
        self.detection_buffer.clear()
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
            "landmark_count": len(self.landmark_map),
            "landmark_map": self.landmark_map,
            "state": self.state,
            "path": self.path,
            "map_update_count": self.map_update_count,
            "map_resolution": self.map_resolution,
            "map_origin": self.map_origin.tolist(),
            "occupancy_grid": self.occupancy_grid,
            "camera_params": self.camera_params,
            "tag_size": self.tag_size,
            "tag_family": self.tag_family,
        }
