"""AprilTag-guided exploration controller for the Go1 robot.

The controller builds a sparse relative map rather than a dense occupancy map.
The robot's startup pose defines map pose (0, 0, 0).  Known-size AprilTags add
metric landmarks and provide the primary odometry whenever they are revisited.
"""

import json
import os
import time
from typing import Any

import cv2
import numpy as np
import torch
from pyapriltags import Detector


# Change this default, or pass goal_tag_id to NavController, for the arena.
DEFAULT_GOAL_TAG_ID = 0

# OpenCV camera axes (right, down, forward) expressed in Go1 body axes
# (forward, left, up).
CAMERA_TO_BODY_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class NavController:
    """Estimate relative pose, map unknown tags, explore, and find a goal tag."""

    def __init__(
        self,
        camera_params: tuple[float, float, float, float],
        tag_size: float = 0.16,
        tag_family: str = "tag36h11",
        goal_tag_id: int = DEFAULT_GOAL_TAG_ID,
        tag_types: dict[int, str] | None = None,
    ):
        """Initialize localization and exploration state.

        Args:
            camera_params: Camera intrinsics (fx, fy, cx, cy).
            tag_size: Length of the AprilTag black square in meters.
            tag_family: AprilTag family used in the arena.
            goal_tag_id: ID of the goal tag.  Its position is not required.
            tag_types: Retained for call-site compatibility.  All non-goal tags
                are handled identically by exploration.
        """
        if isinstance(camera_params, dict):
            camera_params = tuple(camera_params[name] for name in ("fx", "fy", "cx", "cy"))
        if len(camera_params) != 4:
            raise ValueError("camera_params must contain (fx, fy, cx, cy)")

        self.camera_params = tuple(float(value) for value in camera_params)
        self.tag_size = float(tag_size)
        self.tag_family = tag_family
        self.goal_tag_id = int(goal_tag_id)
        self.at_detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Relative pose and uncertainty.  The starting frame is arbitrary but
        # exact by definition; no globally known tag coordinates are needed.
        self.robot_pose = np.zeros(3, dtype=np.float64)
        self.pose_variance = 1.0e-3
        self.yaw_variance = 1.0e-3
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.position_process_variance = 0.04
        self.yaw_process_variance = 0.12
        self.command_speed_scale = 1.0

        # Camera origin relative to the base in body axes (meters).
        self.camera_offset_body = np.array([0.25, 0.0, 0.48], dtype=np.float64)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.gravity_filter_gain = 0.20
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command = np.zeros(3, dtype=np.float64)
        self.has_previous_frame = False
        self.last_sensor_timestamp = None
        self.camera_frame_dt = 0.1
        self.last_pose_source = "initial"

        # Sparse tag map plus one-frame anchors for continuous relative tag
        # odometry.  The global map never directly teleports the robot pose.
        self.landmark_map: dict[int, dict[str, Any]] = {}
        self.previous_tag_anchors: dict[int, dict[str, Any]] = {}
        self.visible_tags: dict[int, dict[str, Any]] = {}
        self.minimum_tag_confidence = 15.0
        self.maximum_tag_position_innovation = 0.45
        self.maximum_tag_yaw_disagreement = np.radians(20.0)
        self.maximum_tag_odometry_position_jump = 0.30
        self.maximum_tag_odometry_yaw_jump = np.radians(30.0)
        self.tag_position_gain = 0.90
        self.tag_yaw_gain = 0.85
        self.tag_map_update_count = 0

        # Goal-tag tracking.  Direct camera-relative control is used while the
        # tag is visible; its mapped position is used briefly for reacquisition.
        self.goal_relative_body = None
        self.goal_world_position = None
        self.robot_body_clearance_radius = 0.28
        self.requested_wall_clearance = 0.60
        self.minimum_surface_clearance = (
            self.robot_body_clearance_radius + self.requested_wall_clearance
        )
        self.preferred_surface_clearance = self.minimum_surface_clearance + 0.10
        self.goal_standoff_distance = self.preferred_surface_clearance
        self.goal_distance_tolerance = 0.08

        # Simple deterministic search: enter open space, approach the first
        # tagged surface, then crab in one fixed direction while facing it.
        # Set this to -1 to reverse the complete wall-following route.
        self.wall_follow_direction = 1
        self.wall_follow_started = False
        self.frames_without_surface = 0
        self.visible_surfaces: list[tuple[np.ndarray, float, np.ndarray]] = []
        self.tracked_surface_world = None
        self.tracked_surface_age = 0
        # Long enough to traverse one small-arena wall after it moves outside
        # the front camera; a newly visible wall immediately replaces it.
        self.tracked_surface_max_age = 200
        self.wall_follow_distance = self.preferred_surface_clearance
        self.wall_follow_forward_speed = 0.22
        self.open_search_speed = 0.28
        self.corner_turn_speed = 0.40
        self.path: list[list[float]] = []

        # Final normalized command limits.
        self.maximum_yaw_speed = 0.45
        self.state = "SEARCH_OPEN"

        self.map_save_interval = 10
        self.last_saved_tag_map_update = 0
        self.map_save_dir = "navigation_maps"
        self._clear_saved_maps()

    # ------------------------------------------------------------------
    # Camera and AprilTag processing
    # ------------------------------------------------------------------
    def detect_apriltags(self, rgb_image: torch.Tensor | np.ndarray, visualize: bool = False) -> dict[int, dict]:
        """Detect tags and return their metric camera-relative poses."""
        try:
            if isinstance(rgb_image, torch.Tensor):
                image = rgb_image.detach().cpu().numpy()
            else:
                image = np.asarray(rgb_image)
            if image.dtype != np.uint8:
                scale = 255.0 if image.size and image.max() <= 1.0 else 1.0
                image = np.clip(image * scale, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            tags = self.at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size,
            )

            detections = {}
            for tag in tags:
                translation = tag.pose_t.reshape(3)
                detections[int(tag.tag_id)] = {
                    "corners": tag.corners.tolist(),
                    "center": tag.center.tolist(),
                    "pose": {
                        "position": translation.tolist(),
                        "rotation_matrix": tag.pose_R.tolist(),
                    },
                    "distance": float(np.linalg.norm(translation)),
                    "confidence": float(tag.decision_margin),
                }
            if visualize:
                self._save_detection_image(image, tags)
            return detections
        except (cv2.error, TypeError, ValueError) as error:
            print(f"[ERROR] AprilTag detection failed: {error}")
            return {}

    def _save_detection_image(self, image: np.ndarray, tags: list) -> None:
        """Save a compact diagnostic image when explicitly requested."""
        try:
            canvas = image.copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
            for tag in tags:
                corners = tag.corners.astype(int)
                for index in range(4):
                    cv2.line(canvas, tuple(corners[index - 1]), tuple(corners[index]), (0, 255, 0), 2)
                center = tag.center.astype(int)
                cv2.putText(
                    canvas,
                    str(tag.tag_id),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
            os.makedirs("apriltag_detection", exist_ok=True)
            filename = os.path.join("apriltag_detection", f"tags_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        except (OSError, cv2.error, ValueError) as error:
            print(f"[WARNING] Failed to save tag image: {error}")

    # ------------------------------------------------------------------
    # Relative pose estimation
    # ------------------------------------------------------------------
    def _roll_pitch_from_gravity(self) -> tuple[float, float]:
        gravity = self.projected_gravity
        pitch = np.arcsin(np.clip(gravity[0], -1.0, 1.0))
        roll = np.arctan2(-gravity[1], -gravity[2])
        return float(roll), float(pitch)

    def _body_rotation_world(self, yaw: float | None = None) -> np.ndarray:
        """Return body-to-map rotation using gravity tilt and estimated yaw."""
        roll, pitch = self._roll_pitch_from_gravity()
        yaw = float(self.robot_pose[2] if yaw is None else yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rotation_z @ rotation_y @ rotation_x

    def _camera_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        rotation_world_from_body = self._body_rotation_world()
        rotation_world_from_camera = rotation_world_from_body @ CAMERA_TO_BODY_ROTATION
        position_world_camera = (
            np.array([self.robot_pose[0], self.robot_pose[1], 0.0])
            + rotation_world_from_body @ self.camera_offset_body
        )
        return position_world_camera, rotation_world_from_camera

    def _predict_pose_without_tags(self, dt: float) -> None:
        """Bridge tag sightings using commanded translation and IMU yaw."""
        delta_body = self.latest_velocity_command[:2] * self.command_speed_scale * dt
        delta_yaw = self.latest_yaw_rate * dt

        previous_yaw = float(self.robot_pose[2])
        cosine, sine = np.cos(previous_yaw), np.sin(previous_yaw)
        body_to_world = np.array([[cosine, -sine], [sine, cosine]])
        self.robot_pose[:2] += body_to_world @ delta_body
        self.robot_pose[2] = _wrap_angle(previous_yaw + delta_yaw)
        self.pose_variance += self.position_process_variance * dt
        self.yaw_variance += self.yaw_process_variance * dt
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.last_pose_source = "imu_command"

    # ------------------------------------------------------------------
    # Sparse AprilTag SLAM
    # ------------------------------------------------------------------
    def _observed_tag_world_pose(self, tag_info: dict) -> tuple[np.ndarray, np.ndarray]:
        translation_camera_from_tag = np.asarray(tag_info["pose"]["position"], dtype=np.float64).reshape(3)
        rotation_camera_from_tag = np.asarray(
            tag_info["pose"]["rotation_matrix"], dtype=np.float64
        ).reshape(3, 3)
        position_world_camera, rotation_world_from_camera = self._camera_pose_world()
        position_world_tag = position_world_camera + rotation_world_from_camera @ translation_camera_from_tag
        rotation_world_from_tag = rotation_world_from_camera @ rotation_camera_from_tag
        return position_world_tag, rotation_world_from_tag

    def _robot_measurement_from_landmark(self, landmark: dict, tag_info: dict) -> np.ndarray:
        """Recover base pose from a stored tag and its current observation."""
        position_world_tag = np.asarray(landmark["position"], dtype=np.float64)
        rotation_world_from_tag = np.asarray(landmark["rotation_matrix"], dtype=np.float64)
        translation_camera_from_tag = np.asarray(tag_info["pose"]["position"], dtype=np.float64).reshape(3)
        rotation_camera_from_tag = np.asarray(
            tag_info["pose"]["rotation_matrix"], dtype=np.float64
        ).reshape(3, 3)

        rotation_world_from_camera = rotation_world_from_tag @ rotation_camera_from_tag.T
        position_world_camera = position_world_tag - rotation_world_from_camera @ translation_camera_from_tag
        rotation_world_from_body = rotation_world_from_camera @ CAMERA_TO_BODY_ROTATION.T
        yaw = _wrap_angle(np.arctan2(rotation_world_from_body[1, 0], rotation_world_from_body[0, 0]))
        position_world_body = position_world_camera - rotation_world_from_body @ self.camera_offset_body
        return np.array([position_world_body[0], position_world_body[1], yaw], dtype=np.float64)

    def _estimate_pose_from_tag_odometry(self, detections: dict[int, dict]) -> bool:
        """Estimate motion from tags shared with the preceding camera frame."""
        measurements = []
        weights = []
        for tag_id, tag_info in detections.items():
            anchor = self.previous_tag_anchors.get(tag_id)
            if anchor is None or float(tag_info.get("confidence", 0.0)) < self.minimum_tag_confidence:
                continue
            confidence = float(tag_info["confidence"])
            distance = float(tag_info["distance"])
            measurement = self._robot_measurement_from_landmark(anchor, tag_info)
            if not np.all(np.isfinite(measurement)):
                continue
            variance = max(1.0e-4, (0.03 + 0.025 * distance * distance) ** 2)
            measurements.append(measurement)
            weights.append(confidence / (variance + float(anchor["variance"])))

        if not measurements:
            return False
        poses = np.asarray(measurements)
        weights_array = np.asarray(weights)

        # A consecutive-frame observation cannot physically move the robot a
        # large distance.  Reject planar-pose flips before consensus/fusion.
        position_innovation = np.linalg.norm(poses[:, :2] - self.robot_pose[:2], axis=1)
        yaw_innovation = np.abs(
            np.array([_wrap_angle(yaw - self.robot_pose[2]) for yaw in poses[:, 2]])
        )
        plausible = (
            position_innovation <= self.maximum_tag_odometry_position_jump
        ) & (yaw_innovation <= self.maximum_tag_odometry_yaw_jump)
        if not np.any(plausible):
            return False
        poses = poses[plausible]
        weights_array = weights_array[plausible]

        # Reject a bad PnP solution or a poorly initialized landmark by using
        # the candidate pose with the strongest nearby consensus as reference.
        consensus_scores = []
        for candidate in poses:
            position_close = np.linalg.norm(poses[:, :2] - candidate[:2], axis=1) <= (
                self.maximum_tag_position_innovation
            )
            yaw_close = np.abs(
                np.array([_wrap_angle(yaw - candidate[2]) for yaw in poses[:, 2]])
            ) <= self.maximum_tag_yaw_disagreement
            consensus_scores.append(float(np.sum(weights_array[position_close & yaw_close])))
        reference = poses[int(np.argmax(consensus_scores))]
        accepted = (
            np.linalg.norm(poses[:, :2] - reference[:2], axis=1)
            <= self.maximum_tag_position_innovation
        ) & (
            np.abs(np.array([_wrap_angle(yaw - reference[2]) for yaw in poses[:, 2]]))
            <= self.maximum_tag_yaw_disagreement
        )
        poses = poses[accepted]
        weights_array = weights_array[accepted]
        weight_sum = float(np.sum(weights_array))
        measured_pose = np.empty(3, dtype=np.float64)
        measured_pose[:2] = np.sum(poses[:, :2] * weights_array[:, None], axis=0) / weight_sum
        measured_pose[2] = np.arctan2(
            np.sum(weights_array * np.sin(poses[:, 2])),
            np.sum(weights_array * np.cos(poses[:, 2])),
        )

        self.robot_pose[:2] += self.tag_position_gain * (measured_pose[:2] - self.robot_pose[:2])
        self.robot_pose[2] = _wrap_angle(
            self.robot_pose[2]
            + self.tag_yaw_gain * _wrap_angle(measured_pose[2] - self.robot_pose[2])
        )
        measurement_variance = max(1.0e-4, 1.0 / weight_sum)
        self.pose_variance = measurement_variance
        self.yaw_variance = max(measurement_variance, np.radians(1.5) ** 2)
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.last_pose_source = "apriltag_relative"
        return True

    def _store_tag_odometry_anchors(self, detections: dict[int, dict]) -> None:
        """Store current tag poses for the next frame's relative odometry."""
        anchors = {}
        for tag_id, tag_info in detections.items():
            confidence = float(tag_info.get("confidence", 0.0))
            if confidence < self.minimum_tag_confidence:
                continue
            try:
                position, rotation = self._observed_tag_world_pose(tag_info)
                distance = float(tag_info["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            anchors[tag_id] = {
                "position": position,
                "rotation_matrix": rotation,
                "variance": max(1.0e-4, (0.03 + 0.025 * distance * distance) ** 2),
            }
        self.previous_tag_anchors = anchors

    def _update_landmark_map(self, detections: dict[int, dict]) -> None:
        """Register new tag anchors and count their later observations."""
        for tag_id, tag_info in detections.items():
            confidence = float(tag_info.get("confidence", 0.0))
            if confidence < self.minimum_tag_confidence:
                continue
            try:
                position, rotation = self._observed_tag_world_pose(tag_info)
            except (KeyError, TypeError, ValueError):
                continue
            distance = float(tag_info["distance"])
            variance = max(1.0e-4, (0.03 + 0.025 * distance * distance) ** 2)
            if tag_id == self.goal_tag_id:
                # Keep reacquisition consistent with the current relative map,
                # even if the original sparse landmark was initialized earlier.
                self.goal_world_position = np.asarray(position[:2]).copy()
            landmark = self.landmark_map.get(tag_id)
            if landmark is None:
                self.landmark_map[tag_id] = {
                    "position": position,
                    "rotation_matrix": rotation,
                    "variance": variance,
                    "observations": 1,
                    "kind": self._tag_kind(tag_id),
                }
            else:
                residual = position - landmark["position"]
                if np.linalg.norm(residual[:2]) > self.maximum_tag_position_innovation:
                    continue
                count = int(landmark["observations"])
                # Keep the world pose fixed: moving an anchor with the same
                # measurement used to localize the robot would reintroduce drift.
                landmark["variance"] = min(float(landmark["variance"]), variance)
                landmark["observations"] = count + 1

            self.tag_map_update_count += 1
        save_due = (
            self.tag_map_update_count - self.last_saved_tag_map_update >= self.map_save_interval
        )
        if save_due:
            self._save_sparse_tag_map()
            self.last_saved_tag_map_update = self.tag_map_update_count

    # ------------------------------------------------------------------
    # Simple wall following and goal approach
    # ------------------------------------------------------------------
    def _tag_kind(self, tag_id: int) -> str:
        return "goal" if tag_id == self.goal_tag_id else "landmark"

    @staticmethod
    def _surface_geometry(
        point_body_3d: np.ndarray, normal_body: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray] | None:
        """Return tag point, plane distance, and planar direction to the wall."""
        point_body_3d = np.asarray(point_body_3d, dtype=np.float64).reshape(3)
        normal_body = np.asarray(normal_body, dtype=np.float64).reshape(3)
        point_body = point_body_3d[:2]
        point_range = float(np.linalg.norm(point_body))
        normal_xy = normal_body[:2]
        normal_xy_norm = float(np.linalg.norm(normal_xy))
        if point_range <= 0.05:
            return None
        if normal_xy_norm < 0.45:
            return point_body, point_range, point_body / point_range
        normal_body /= max(float(np.linalg.norm(normal_body)), 1.0e-6)
        normal_xy = normal_body[:2] / np.linalg.norm(normal_body[:2])
        sign = 1.0 if np.dot(normal_xy, point_body) >= 0.0 else -1.0
        toward_surface = sign * normal_xy
        distance = min(abs(float(np.dot(normal_body, point_body_3d))), point_range)
        return point_body, distance, toward_surface

    def _update_visible_surfaces(self, detections: dict[int, dict]) -> None:
        """Extract current wall planes and retain the closest one briefly."""
        surfaces = []
        world_surfaces = []
        for tag_info in detections.values():
            if float(tag_info.get("confidence", 0.0)) < 0.5 * self.minimum_tag_confidence:
                continue
            try:
                translation = np.asarray(
                    tag_info["pose"]["position"], dtype=np.float64
                ).reshape(3)
                rotation_camera_from_tag = np.asarray(
                    tag_info["pose"]["rotation_matrix"], dtype=np.float64
                ).reshape(3, 3)
                point_body_3d = (
                    self.camera_offset_body + CAMERA_TO_BODY_ROTATION @ translation
                )
                normal_body = CAMERA_TO_BODY_ROTATION @ rotation_camera_from_tag[:, 2]
                surface = self._surface_geometry(point_body_3d, normal_body)
                position_world, rotation_world_from_tag = self._observed_tag_world_pose(tag_info)
            except (KeyError, TypeError, ValueError):
                continue
            if surface is None or not np.all(np.isfinite(surface[0])):
                continue
            surfaces.append(surface)
            world_surfaces.append(
                {
                    "position": position_world,
                    "normal": rotation_world_from_tag[:, 2],
                }
            )
        self.visible_surfaces = surfaces
        if surfaces:
            closest = int(np.argmin([surface[1] for surface in surfaces]))
            self.tracked_surface_world = world_surfaces[closest]
            self.tracked_surface_age = 0
        else:
            self.tracked_surface_age += 1

    def _visible_surface(self) -> tuple[float, float, np.ndarray] | None:
        """Return distance, bearing, and direction toward the tracked wall."""
        if self.visible_surfaces:
            point, distance, toward_surface = min(
                self.visible_surfaces, key=lambda surface: surface[1]
            )
            return distance, float(np.arctan2(point[1], point[0])), toward_surface
        if self.tracked_surface_world is None or self.tracked_surface_age > self.tracked_surface_max_age:
            return None

        rotation_world_from_body = self._body_rotation_world()
        rotation_body_from_world = rotation_world_from_body.T
        position_world_body = np.array(
            [self.robot_pose[0], self.robot_pose[1], 0.0], dtype=np.float64
        )
        point_body_3d = rotation_body_from_world @ (
            np.asarray(self.tracked_surface_world["position"]) - position_world_body
        )
        normal_body = rotation_body_from_world @ np.asarray(
            self.tracked_surface_world["normal"]
        )
        surface = self._surface_geometry(point_body_3d, normal_body)
        if surface is None:
            return None
        point, distance, toward_surface = surface
        return distance, float(np.arctan2(point[1], point[0])), toward_surface

    def _apply_surface_clearance(self, command: np.ndarray) -> np.ndarray:
        """Gently drift away and enforce the minimum tag-surface distance."""
        surface = self._visible_surface()
        adjusted = np.asarray(command, dtype=np.float64).copy()
        adjusted[1] = 0.0
        if surface is None:
            return adjusted
        distance, bearing, toward_surface = surface
        if distance >= self.preferred_surface_clearance:
            return adjusted

        toward_speed = float(adjusted[0] * toward_surface[0])
        if toward_speed > 0.0:
            scale = np.clip(
                (distance - self.minimum_surface_clearance)
                / (self.preferred_surface_clearance - self.minimum_surface_clearance),
                0.0,
                1.0,
            )
            adjusted[0] *= scale
        if distance < self.minimum_surface_clearance:
            adjusted[0] = -0.16 if abs(bearing) < 0.70 else 0.0
            adjusted[2] = self.corner_turn_speed * self.wall_follow_direction
            self.state = "SURFACE_BACKOFF"
        return adjusted

    def _goal_command(self) -> np.ndarray | None:
        """Approach a visible goal tag and stop before its wall."""
        if self.goal_relative_body is None:
            return None
        target = np.asarray(self.goal_relative_body, dtype=np.float64)
        distance = float(np.linalg.norm(target))
        bearing = float(np.arctan2(target[1], target[0]))
        radial_error = distance - self.goal_standoff_distance
        if radial_error <= self.goal_distance_tolerance and abs(bearing) < 0.15:
            self.state = "ARRIVED"
            return np.zeros(3, dtype=np.float32)

        alignment = max(0.0, float(np.cos(bearing)))
        forward = np.clip(0.70 * radial_error * alignment, 0.0, 0.35)
        yaw = np.clip(0.90 * bearing, -self.maximum_yaw_speed, self.maximum_yaw_speed)
        self.state = "APPROACH_GOAL"
        return np.array([forward, 0.0, yaw], dtype=np.float32)

    def _exploration_command(self) -> np.ndarray:
        """Enter open space once, then follow every tagged wall consistently."""
        surface = self._visible_surface()
        if surface is None:
            self.frames_without_surface += 1
            if not self.wall_follow_started:
                self.state = "SEARCH_OPEN"
                return np.array(
                    [self.open_search_speed, 0.0, 0.08 * self.wall_follow_direction],
                    dtype=np.float32,
                )

            # Tags normally disappear at a corner.  Keep turning in exactly the
            # same direction until tags on the next wall enter the camera.
            self.state = "TURN_CORNER"
            forward = 0.0 if self.frames_without_surface <= 12 else 0.08
            return np.array(
                [forward, 0.0, self.corner_turn_speed * self.wall_follow_direction],
                dtype=np.float32,
            )

        self.frames_without_surface = 0
        distance, bearing, toward_surface = surface

        if not self.wall_follow_started:
            if distance > self.wall_follow_distance + 0.08:
                forward = np.clip(
                    0.65 * (distance - self.wall_follow_distance), 0.08, 0.30
                )
                forward *= max(0.0, float(np.cos(bearing)))
                yaw = np.clip(
                    1.10 * bearing, -self.maximum_yaw_speed, self.maximum_yaw_speed
                )
                self.state = "APPROACH_WALL"
                return np.array([forward, 0.0, yaw], dtype=np.float32)
            self.wall_follow_started = True

        # Build a forward-only desired heading from a fixed wall tangent plus a
        # small distance correction.  The robot turns first, then walks ahead.
        tangent = self.wall_follow_direction * np.array(
            [-toward_surface[1], toward_surface[0]], dtype=np.float64
        )
        distance_correction = np.clip(
            0.55 * (distance - self.wall_follow_distance), -0.16, 0.16
        )
        desired_direction = (
            self.wall_follow_forward_speed * tangent
            + distance_correction * toward_surface
        )
        heading_error = float(
            np.arctan2(desired_direction[1], desired_direction[0])
        )
        yaw = np.clip(
            1.20 * heading_error, -self.maximum_yaw_speed, self.maximum_yaw_speed
        )
        forward = np.linalg.norm(desired_direction) * max(
            0.0, float(np.cos(heading_error))
        )
        self.state = "FOLLOW_WALL"
        return np.array([forward, 0.0, yaw], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public controller API
    # ------------------------------------------------------------------
    def update(self, observations: dict[str, Any]) -> None:
        """Update pose, sparse tag map, goal observation, and wall tracking."""
        angular_velocity = observations.get("base_ang_vel")
        if angular_velocity is not None:
            angular_velocity = np.asarray(angular_velocity, dtype=np.float64).reshape(-1)
            if angular_velocity.size >= 3:
                self.latest_yaw_rate = float(angular_velocity[2])

        gravity = observations.get("projected_gravity")
        if gravity is not None:
            gravity = np.asarray(gravity, dtype=np.float64).reshape(-1)[:3]
            norm = float(np.linalg.norm(gravity))
            if gravity.size == 3 and np.all(np.isfinite(gravity)) and norm > 1.0e-6:
                gravity /= norm
                gain = self.gravity_filter_gain
                self.projected_gravity = (1.0 - gain) * self.projected_gravity + gain * gravity
                self.projected_gravity /= np.linalg.norm(self.projected_gravity)

        command = observations.get("velocity_commands")
        if command is not None:
            command = np.asarray(command, dtype=np.float64).reshape(-1)
            if command.size >= 3:
                self.latest_velocity_command = command[:3].copy()

        rgb_image = observations.get("camera_rgb")
        if rgb_image is None:
            return
        if isinstance(rgb_image, torch.Tensor):
            image = rgb_image.detach().cpu().numpy()
        else:
            image = np.asarray(rgb_image)
        if image.dtype != np.uint8:
            scale = 255.0 if image.size and image.max() <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        sensor_timestamp = observations.get("timestamp")
        if not self.has_previous_frame:
            dt = 0.0
        elif sensor_timestamp is not None and self.last_sensor_timestamp is not None:
            dt = float(np.clip(float(sensor_timestamp) - self.last_sensor_timestamp, 1.0e-3, 0.5))
        else:
            dt = self.camera_frame_dt
        if dt > 0.0:
            self._predict_pose_without_tags(dt)

        detections = self.detect_apriltags(image)
        self._estimate_pose_from_tag_odometry(detections)
        self._update_landmark_map(detections)
        self._store_tag_odometry_anchors(detections)
        self.visible_tags = detections

        goal = detections.get(self.goal_tag_id)
        if goal is not None:
            translation = np.asarray(goal["pose"]["position"], dtype=np.float64)
            self.goal_relative_body = np.array(
                [translation[2] + self.camera_offset_body[0], -translation[0]], dtype=np.float64
            )
        else:
            self.goal_relative_body = None

        self._update_visible_surfaces(detections)
        tracked_surface = self._visible_surface()
        free_surface_clearance = (
            None
            if tracked_surface is None
            else tracked_surface[0] - self.robot_body_clearance_radius
        )
        clearance_text = (
            "unknown" if free_surface_clearance is None else f"{free_surface_clearance:.2f} m"
        )
        if not self.path or np.linalg.norm(self.robot_pose[:2] - np.asarray(self.path[-1])) >= 0.05:
            self.path.append(self.robot_pose[:2].copy().tolist())
            self.path = self.path[-1000:]

        self.has_previous_frame = True
        self.last_sensor_timestamp = None if sensor_timestamp is None else float(sensor_timestamp)
        print(
            "[NavController] Pose "
            f"x={self.robot_pose[0]:.3f}, y={self.robot_pose[1]:.3f}, "
            f"yaw={np.degrees(self.robot_pose[2]):.1f} deg; "
            f"source={self.last_pose_source}, tags={sorted(detections)}, "
            f"goal_visible={goal is not None}, "
            f"surface_clearance={clearance_text}"
        )

    def get_command(self) -> np.ndarray:
        """Return normalized body-frame ``[vx, vy, yaw_rate]``."""
        goal_command = self._goal_command()
        if goal_command is not None:
            command = goal_command
        else:
            command = self._exploration_command()

        command = self._apply_surface_clearance(command)
        return np.clip(command, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """Reset runtime state without deleting saved maps from this run."""
        self.robot_pose.fill(0.0)
        self.pose_variance = 1.0e-3
        self.yaw_variance = 1.0e-3
        self.pose_covariance = np.diag([self.pose_variance, self.pose_variance, self.yaw_variance])
        self.projected_gravity[:] = (0.0, 0.0, -1.0)
        self.latest_yaw_rate = 0.0
        self.latest_velocity_command.fill(0.0)
        self.has_previous_frame = False
        self.last_sensor_timestamp = None
        self.last_pose_source = "initial"
        self.landmark_map.clear()
        self.previous_tag_anchors.clear()
        self.visible_tags.clear()
        self.visible_surfaces.clear()
        self.tracked_surface_world = None
        self.tracked_surface_age = 0
        self.goal_relative_body = None
        self.goal_world_position = None
        self.path.clear()
        self.frames_without_surface = 0
        self.wall_follow_started = False
        self.tag_map_update_count = 0
        self.last_saved_tag_map_update = 0
        self.state = "SEARCH_OPEN"

    def get_debug_info(self) -> dict[str, Any]:
        """Return compact state for logging and visualization."""
        landmarks = {
            tag_id: {
                "position": np.asarray(data["position"]).tolist(),
                "rotation_matrix": np.asarray(data["rotation_matrix"]).tolist(),
                "variance": float(data["variance"]),
                "observations": int(data["observations"]),
                "kind": data["kind"],
            }
            for tag_id, data in self.landmark_map.items()
        }
        return {
            "robot_pose": self.robot_pose.tolist(),
            "pose_covariance": self.pose_covariance.tolist(),
            "pose_source": self.last_pose_source,
            "surface_clearance": (
                None
                if self._visible_surface() is None
                else float(self._visible_surface()[0]) - self.robot_body_clearance_radius
            ),
            "state": self.state,
            "goal_tag_id": self.goal_tag_id,
            "goal_visible": self.goal_relative_body is not None,
            "goal_world_position": (
                None if self.goal_world_position is None else self.goal_world_position.tolist()
            ),
            "visible_tag_ids": sorted(self.visible_tags),
            "landmark_map": landmarks,
            "wall_follow_started": self.wall_follow_started,
            "wall_follow_direction": self.wall_follow_direction,
            "path": self.path,
        }

    # ------------------------------------------------------------------
    # Sparse-map persistence
    # ------------------------------------------------------------------
    def _clear_saved_maps(self) -> None:
        try:
            os.makedirs(self.map_save_dir, exist_ok=True)
            for filename in os.listdir(self.map_save_dir):
                generated_prefix = filename.startswith(("tag_map_", "occupancy_map_"))
                if generated_prefix and filename.endswith((".json", ".png", ".npy", ".npz")):
                    os.remove(os.path.join(self.map_save_dir, filename))
        except OSError as error:
            print(f"[WARNING] Failed to clear tag maps: {error}")

    def _save_sparse_tag_map(self) -> None:
        """Save landmark data and a simple top-down visualization."""
        try:
            os.makedirs(self.map_save_dir, exist_ok=True)
            stem = os.path.join(self.map_save_dir, f"tag_map_{self.tag_map_update_count:06d}")
            serializable_landmarks = {
                str(tag_id): {
                    "position": np.asarray(data["position"]).tolist(),
                    "rotation_matrix": np.asarray(data["rotation_matrix"]).tolist(),
                    "variance": float(data["variance"]),
                    "observations": int(data["observations"]),
                    "kind": data["kind"],
                }
                for tag_id, data in self.landmark_map.items()
            }
            with open(f"{stem}.json", "w", encoding="utf-8") as map_file:
                json.dump(
                    {
                        "robot_pose": self.robot_pose.tolist(),
                        "goal_tag_id": self.goal_tag_id,
                        "landmarks": serializable_landmarks,
                    },
                    map_file,
                    indent=2,
                )

            canvas_size = 600
            pixels_per_meter = 50.0
            center = canvas_size // 2
            canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)

            def to_pixel(point: np.ndarray) -> tuple[int, int]:
                point = np.asarray(point)
                return int(center + pixels_per_meter * point[0]), int(center - pixels_per_meter * point[1])

            for first, second in zip(self.path[:-1], self.path[1:]):
                cv2.line(canvas, to_pixel(first), to_pixel(second), (180, 180, 180), 1)
            colors = {"landmark": (30, 90, 200), "goal": (0, 0, 255)}
            for tag_id, data in self.landmark_map.items():
                pixel = to_pixel(data["position"][:2])
                color = colors.get(data["kind"], (100, 60, 160))
                cv2.circle(canvas, pixel, 5, color, -1)
                cv2.putText(
                    canvas,
                    str(tag_id),
                    (pixel[0] + 6, pixel[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            robot_pixel = to_pixel(self.robot_pose[:2])
            cv2.circle(canvas, robot_pixel, 6, (0, 180, 0), -1)
            heading_pixel = to_pixel(
                self.robot_pose[:2]
                + 0.35 * np.array([np.cos(self.robot_pose[2]), np.sin(self.robot_pose[2])])
            )
            cv2.line(canvas, robot_pixel, heading_pixel, (0, 120, 0), 2)
            if not cv2.imwrite(f"{stem}.png", canvas):
                raise OSError("cv2.imwrite returned false")
            print(f"[NavController] Saved sparse tag map at {stem}")
        except (OSError, TypeError, ValueError, cv2.error) as error:
            print(f"[WARNING] Failed to save sparse tag map: {error}")
