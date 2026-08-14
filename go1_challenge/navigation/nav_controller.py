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

# Optional per-tag overrides.  By default, non-goal IDs below 20 are walls,
# IDs above 20 are obstacles, and ID 20 is generic.
DEFAULT_TAG_TYPES: dict[int, str] = {}

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
            tag_types: Optional mapping from tag ID to ``wall``, ``obstacle``,
                or ``goal``.  Unlisted non-goal tags are treated conservatively
                as generic obstacles.
        """
        if isinstance(camera_params, dict):
            camera_params = tuple(camera_params[name] for name in ("fx", "fy", "cx", "cy"))
        if len(camera_params) != 4:
            raise ValueError("camera_params must contain (fx, fy, cx, cy)")

        self.camera_params = tuple(float(value) for value in camera_params)
        self.tag_size = float(tag_size)
        self.tag_family = tag_family
        self.goal_tag_id = int(goal_tag_id)
        self.tag_types = dict(DEFAULT_TAG_TYPES if tag_types is None else tag_types)
        self.tag_types[self.goal_tag_id] = "goal"
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

        # Sparse tag map.  Each tag stores a full map-frame pose, uncertainty,
        # observation count, and semantic type.
        self.landmark_map: dict[int, dict[str, Any]] = {}
        self.visible_tags: dict[int, dict[str, Any]] = {}
        self.minimum_tag_confidence = 15.0
        self.maximum_tag_position_innovation = 0.45
        self.maximum_tag_yaw_disagreement = np.radians(20.0)
        self.tag_position_gain = 0.90
        self.tag_yaw_gain = 0.85
        self.tag_map_update_count = 0

        # Goal-tag tracking.  Direct camera-relative control is used while the
        # tag is visible; its mapped position is used briefly for reacquisition.
        self.goal_relative_body = None
        self.goal_world_position = None
        self.goal_standoff_distance = 0.55
        self.goal_distance_tolerance = 0.08

        # Tag-space gap exploration.  A tag blocks an angular interval based on
        # range and a conservative semantic safety radius.
        self.sector_count = 41
        self.sector_angles = np.linspace(-0.5, 0.5, self.sector_count)
        self.sector_clearance = np.full(self.sector_count, 3.0, dtype=np.float64)
        self.maximum_tag_avoidance_range = 2.0
        self.emergency_tag_distance = 0.35
        self.front_blocking_distance = 0.75
        self.wall_safety_radius = 0.55
        self.obstacle_safety_radius = 0.50
        self.generic_safety_radius = 0.42
        self.best_gap_angle = 0.0
        self.preferred_turn_side = 1

        # Exploration memory prevents repeated loops without maintaining a
        # dense grid.  Positions are counted in coarse 0.5 m cells.
        self.visit_cell_size = 0.50
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.heading_visit_counts = np.zeros(24, dtype=np.int32)
        self.path: list[list[float]] = []
        self.recent_pose_history: list[list[float]] = []
        self.seen_tag_ids: set[int] = set()
        self.frames_without_tags = 0
        self.frames_since_new_tag = 0
        self.control_step = 0
        self.initial_scan_steps = 15
        self.recovery_steps_remaining = 0
        self.recovery_duration_steps = 8
        self.no_tag_scan_threshold = 5
        self.stagnation_window = 20
        self.stagnation_distance = 0.12

        # Final normalized command limits.  Exploration favors forward motion;
        # lateral motion stays small because the monocular camera faces forward.
        self.maximum_forward_speed = 0.50
        self.maximum_lateral_speed = 0.22
        self.maximum_yaw_speed = 0.45
        self.yaw_command_gain = 0.85
        self.state = "INITIAL_SCAN"

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

    def _correct_pose_from_known_landmarks(self, detections: dict[int, dict]) -> bool:
        """Use reobserved tags as the primary robot-pose measurement."""
        measurements = []
        weights = []
        for tag_id, tag_info in detections.items():
            landmark = self.landmark_map.get(tag_id)
            if landmark is None or float(tag_info.get("confidence", 0.0)) < self.minimum_tag_confidence:
                continue
            confidence = float(tag_info["confidence"])
            distance = float(tag_info["distance"])
            measurement = self._robot_measurement_from_landmark(landmark, tag_info)
            if not np.all(np.isfinite(measurement)):
                continue
            variance = max(1.0e-4, (0.03 + 0.025 * distance * distance) ** 2)
            measurements.append(measurement)
            weights.append(confidence / (variance + float(landmark["variance"])))

        if not measurements:
            return False
        poses = np.asarray(measurements)
        weights_array = np.asarray(weights)

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
        self.last_pose_source = "apriltag"
        return True

    def _update_landmark_map(self, detections: dict[int, dict]) -> None:
        """Register new tag anchors and count their later observations."""
        new_tag_seen = False
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
            landmark = self.landmark_map.get(tag_id)
            if landmark is None:
                new_tag_seen = True
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
            self.seen_tag_ids.add(tag_id)
            if tag_id == self.goal_tag_id:
                self.goal_world_position = np.asarray(self.landmark_map[tag_id]["position"][:2]).copy()

        self.frames_since_new_tag = 0 if new_tag_seen else self.frames_since_new_tag + 1
        save_due = (
            self.tag_map_update_count - self.last_saved_tag_map_update >= self.map_save_interval
        )
        if save_due:
            self._save_sparse_tag_map()
            self.last_saved_tag_map_update = self.tag_map_update_count

    # ------------------------------------------------------------------
    # Gap exploration and goal approach
    # ------------------------------------------------------------------
    def _tag_kind(self, tag_id: int) -> str:
        if tag_id == self.goal_tag_id:
            return "goal"
        if tag_id in self.tag_types:
            return self.tag_types[tag_id]
        if tag_id < 20:
            return "wall"
        if tag_id > 20:
            return "obstacle"
        return "generic"

    def _tag_safety_radius(self, tag_id: int) -> float:
        kind = self._tag_kind(tag_id)
        if kind == "wall":
            return self.wall_safety_radius
        if kind == "obstacle":
            return self.obstacle_safety_radius
        return self.generic_safety_radius

    def _update_visible_tag_sectors(self, detections: dict[int, dict], image_width: int) -> None:
        """Convert visible and nearby mapped tags into blocked angular sectors."""
        fx, _, cx, _ = self.camera_params
        half_fov = max(0.10, np.arctan2(max(cx, image_width - cx), fx))
        self.sector_angles = np.linspace(-half_fov, half_fov, self.sector_count)
        self.sector_clearance.fill(self.maximum_tag_avoidance_range)

        obstacles = []
        for tag_id, tag_info in detections.items():
            if tag_id == self.goal_tag_id:
                continue
            translation = np.asarray(tag_info["pose"]["position"], dtype=np.float64)
            point_body = np.array(
                [translation[2] + self.camera_offset_body[0], -translation[0]], dtype=np.float64
            )
            obstacles.append((tag_id, point_body))

        # Retain memory of close side obstacles even after they leave the
        # forward camera.  This limits unsafe lateral commands into unseen space.
        yaw = float(self.robot_pose[2])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        world_to_body = np.array([[cosine, sine], [-sine, cosine]])
        for tag_id, landmark in self.landmark_map.items():
            if tag_id in detections or tag_id == self.goal_tag_id:
                continue
            point_body = world_to_body @ (np.asarray(landmark["position"][:2]) - self.robot_pose[:2])
            if np.linalg.norm(point_body) < 1.25:
                obstacles.append((tag_id, point_body))

        for tag_id, point_body in obstacles:
            distance = float(np.linalg.norm(point_body))
            if distance <= 0.05 or distance > self.maximum_tag_avoidance_range:
                continue
            bearing = float(np.arctan2(point_body[1], point_body[0]))
            if abs(bearing) > half_fov + 0.35:
                continue
            angular_radius = np.arctan2(self._tag_safety_radius(tag_id), max(distance, 0.10))
            blocked = np.abs(self.sector_angles - bearing) <= angular_radius
            self.sector_clearance[blocked] = np.minimum(self.sector_clearance[blocked], distance)

    def _visit_key(self, point: np.ndarray) -> tuple[int, int]:
        return tuple(np.floor(np.asarray(point) / self.visit_cell_size).astype(int))

    def _candidate_visit_penalty(self, angle_body: float) -> float:
        heading_world = self.robot_pose[2] + angle_body
        endpoint = self.robot_pose[:2] + np.array([np.cos(heading_world), np.sin(heading_world)])
        cell_penalty = self.visit_counts.get(self._visit_key(endpoint), 0)
        heading_bin = int((heading_world % (2.0 * np.pi)) / (2.0 * np.pi) * len(self.heading_visit_counts))
        return 0.20 * min(cell_penalty, 10) + 0.04 * min(int(self.heading_visit_counts[heading_bin]), 30)

    def _select_gap(self, preferred_angle: float | None = None) -> float | None:
        """Return the safest low-visit heading inside the best visible gap."""
        free = self.sector_clearance >= self.front_blocking_distance
        gaps = []
        start = None
        for index, blocked in enumerate(np.append(~free, True)):
            if not blocked and start is None:
                start = index
            elif blocked and start is not None:
                if index - start >= 2:
                    gaps.append((start, index - 1))
                start = None
        if not gaps:
            return None

        angular_step = abs(float(self.sector_angles[1] - self.sector_angles[0]))
        best_angle = 0.0
        best_score = -np.inf
        for gap_start, gap_end in gaps:
            width = float(self.sector_angles[gap_end] - self.sector_angles[gap_start] + angular_step)
            mean_clearance = float(np.mean(self.sector_clearance[gap_start : gap_end + 1]))
            for index in range(gap_start, gap_end + 1):
                angle = float(self.sector_angles[index])
                preferred_error = 0.0 if preferred_angle is None else abs(_wrap_angle(angle - preferred_angle))
                score = (
                    2.0 * width
                    + 0.55 * mean_clearance
                    - 0.45 * abs(angle)
                    - 0.90 * preferred_error
                    - self._candidate_visit_penalty(angle)
                )
                if score > best_score:
                    best_score = score
                    best_angle = angle
        self.best_gap_angle = best_angle
        return best_angle

    def _stagnating(self) -> bool:
        if len(self.recent_pose_history) < self.stagnation_window:
            return False
        recent = np.asarray(self.recent_pose_history[-self.stagnation_window :])
        return float(np.linalg.norm(recent[-1] - recent[0])) < self.stagnation_distance

    def _goal_command(self) -> np.ndarray | None:
        """Approach a visible goal tag and stop before its wall."""
        if self.goal_relative_body is None:
            return None
        target = np.asarray(self.goal_relative_body, dtype=np.float64)
        distance = float(np.linalg.norm(target))
        bearing = float(np.arctan2(target[1], target[0]))
        goal_sector = int(np.argmin(np.abs(self.sector_angles - bearing)))
        if self.sector_clearance[goal_sector] < self.front_blocking_distance:
            # Another tag-marked wall/obstacle blocks the direct goal corridor.
            # Let gap exploration route around it while retaining goal bias.
            return None
        radial_error = distance - self.goal_standoff_distance
        if radial_error <= self.goal_distance_tolerance and abs(bearing) < 0.15:
            self.state = "ARRIVED"
            return np.zeros(3, dtype=np.float32)

        direction = target / max(distance, 1.0e-6)
        forward = np.clip(0.70 * radial_error * direction[0], 0.0, 0.35)
        lateral = np.clip(0.50 * radial_error * direction[1], -0.18, 0.18)
        yaw = np.clip(0.90 * bearing, -self.maximum_yaw_speed, self.maximum_yaw_speed)
        self.state = "APPROACH_GOAL"
        return np.array([forward, lateral, yaw], dtype=np.float32)

    def _exploration_command(self) -> np.ndarray:
        """Explore visible gaps, with scan and anti-stagnation recovery."""
        if self.control_step < self.initial_scan_steps:
            self.state = "INITIAL_SCAN"
            return np.array([0.0, 0.0, 0.35], dtype=np.float32)
        if self.control_step == self.initial_scan_steps:
            self.recent_pose_history.clear()

        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining == 0:
                self.recent_pose_history.clear()
            self.state = "RECOVERY"
            return np.array([0.0, 0.0, self.maximum_yaw_speed * self.preferred_turn_side], dtype=np.float32)

        if self._stagnating() and self.frames_since_new_tag > 5:
            self.preferred_turn_side *= -1
            self.recovery_steps_remaining = self.recovery_duration_steps
            self.recent_pose_history.clear()
            self.state = "RECOVERY"
            return np.array([0.0, 0.0, self.maximum_yaw_speed * self.preferred_turn_side], dtype=np.float32)

        if self.frames_without_tags >= self.no_tag_scan_threshold:
            self.state = "SEARCH_ROTATE"
            return np.array([0.0, 0.0, 0.35 * self.preferred_turn_side], dtype=np.float32)

        preferred_angle = None
        if self.goal_world_position is not None:
            goal_delta = self.goal_world_position - self.robot_pose[:2]
            preferred_angle = _wrap_angle(np.arctan2(goal_delta[1], goal_delta[0]) - self.robot_pose[2])
            if self.goal_relative_body is None and np.linalg.norm(goal_delta) < 0.8:
                self.state = "REACQUIRE_GOAL"
                return np.array([0.0, 0.0, 0.30 * np.sign(preferred_angle or 1.0)], dtype=np.float32)

        gap_angle = self._select_gap(preferred_angle)
        if gap_angle is None:
            self.state = "TURN_TO_GAP"
            return np.array([0.0, 0.0, self.maximum_yaw_speed * self.preferred_turn_side], dtype=np.float32)

        center_index = int(np.argmin(np.abs(self.sector_angles)))
        front_clearance = float(self.sector_clearance[center_index])
        if front_clearance < self.emergency_tag_distance:
            self.preferred_turn_side = 1 if gap_angle >= 0.0 else -1
            self.state = "EMERGENCY_AVOID"
            return np.array([0.0, 0.15 * self.preferred_turn_side, 0.45 * self.preferred_turn_side], dtype=np.float32)

        alignment = max(0.25, np.cos(gap_angle))
        clearance_factor = np.clip(
            (front_clearance - self.emergency_tag_distance)
            / (self.front_blocking_distance - self.emergency_tag_distance),
            0.25,
            1.0,
        )
        forward = self.maximum_forward_speed * alignment * clearance_factor
        lateral = np.clip(0.20 * np.sin(gap_angle), -self.maximum_lateral_speed, self.maximum_lateral_speed)
        yaw = np.clip(
            self.yaw_command_gain * gap_angle,
            -self.maximum_yaw_speed,
            self.maximum_yaw_speed,
        )
        self.state = "EXPLORE_GAP"
        return np.array([forward, lateral, yaw], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public controller API
    # ------------------------------------------------------------------
    def update(self, observations: dict[str, Any]) -> None:
        """Update pose, sparse tag map, goal observation, and local gaps."""
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
        self._correct_pose_from_known_landmarks(detections)
        self._update_landmark_map(detections)
        self.visible_tags = detections
        if detections:
            self.frames_without_tags = 0
        else:
            self.frames_without_tags += 1

        goal = detections.get(self.goal_tag_id)
        if goal is not None:
            translation = np.asarray(goal["pose"]["position"], dtype=np.float64)
            self.goal_relative_body = np.array(
                [translation[2] + self.camera_offset_body[0], -translation[0]], dtype=np.float64
            )
        else:
            self.goal_relative_body = None

        self._update_visible_tag_sectors(detections, image.shape[1])
        visit_key = self._visit_key(self.robot_pose[:2])
        self.visit_counts[visit_key] = self.visit_counts.get(visit_key, 0) + 1
        self.recent_pose_history.append(self.robot_pose[:2].copy().tolist())
        self.recent_pose_history = self.recent_pose_history[-self.stagnation_window :]
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
            f"goal_visible={goal is not None}"
        )

    def get_command(self) -> np.ndarray:
        """Return normalized body-frame ``[vx, vy, yaw_rate]``."""
        self.control_step += 1
        goal_command = self._goal_command()
        if goal_command is not None:
            command = goal_command
        else:
            command = self._exploration_command()

        heading_world = self.robot_pose[2] + np.arctan2(command[1], max(command[0], 1.0e-3))
        heading_bin = int((heading_world % (2.0 * np.pi)) / (2.0 * np.pi) * len(self.heading_visit_counts))
        self.heading_visit_counts[heading_bin] += 1
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
        self.visible_tags.clear()
        self.goal_relative_body = None
        self.goal_world_position = None
        self.sector_clearance.fill(self.maximum_tag_avoidance_range)
        self.best_gap_angle = 0.0
        self.visit_counts.clear()
        self.heading_visit_counts.fill(0)
        self.path.clear()
        self.recent_pose_history.clear()
        self.seen_tag_ids.clear()
        self.frames_without_tags = 0
        self.frames_since_new_tag = 0
        self.control_step = 0
        self.recovery_steps_remaining = 0
        self.preferred_turn_side = 1
        self.tag_map_update_count = 0
        self.last_saved_tag_map_update = 0
        self.state = "INITIAL_SCAN"

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
            "state": self.state,
            "goal_tag_id": self.goal_tag_id,
            "goal_visible": self.goal_relative_body is not None,
            "goal_world_position": (
                None if self.goal_world_position is None else self.goal_world_position.tolist()
            ),
            "visible_tag_ids": sorted(self.visible_tags),
            "landmark_map": landmarks,
            "best_gap_angle": self.best_gap_angle,
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
            colors = {"wall": (180, 80, 20), "obstacle": (20, 20, 20), "goal": (0, 0, 255)}
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
