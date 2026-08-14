# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def finite_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    height_tolerance: float = 0.0,
    max_height_error: float = 1.0,
) -> torch.Tensor:
    """Penalize terrain-relative base height outside a tolerance band."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    root_height = asset.data.root_pos_w.torch[:, 2]
    ray_heights = sensor.data.ray_hits_w.torch[..., 2]
    finite_hits = torch.isfinite(ray_heights)
    hit_count = finite_hits.sum(dim=1)
    safe_hits = torch.where(finite_hits, ray_heights, 0.0)
    terrain_height = safe_hits.sum(dim=1) / hit_count.clamp_min(1)
    terrain_relative_height = root_height - terrain_height

    # Keep the unweighted physical measurement available to the training logger.
    # NaN marks environments for which the scanner did not hit any terrain so
    # those samples can be excluded from the reported average.
    env._last_base_height_above_terrain = torch.where(
        hit_count > 0,
        terrain_relative_height,
        torch.full_like(terrain_relative_height, torch.nan),
    ).detach()
    env._base_height_target = target_height

    height_error = terrain_relative_height - target_height
    height_error = torch.where(hit_count > 0, height_error, torch.zeros_like(height_error))
    height_error = torch.nan_to_num(height_error, nan=0.0)

    # Allow natural vertical body motion inside the dead band. Only the
    # distance beyond the tolerance contributes to the quadratic penalty.
    excess_height_error = torch.clamp(
        torch.abs(height_error) - height_tolerance, min=0.0, max=max_height_error
    )
    return torch.square(excess_height_error)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the default joint pose only for near-zero commands.

    Unlike Isaac Lab stock stand-still term, the zero-command test includes
    yaw rate as well as planar velocity so the penalty does not oppose commanded
    in-place turns.
    """
    asset = env.scene[asset_cfg.name]
    joint_deviation = torch.abs(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    command = env.command_manager.get_command(command_name)
    command_is_zero = torch.linalg.vector_norm(command, dim=1) < command_threshold
    return torch.sum(joint_deviation, dim=1) * command_is_zero


def prolonged_foot_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_stance_time: float,
    max_stance_time: float,
    velocity_gain: float,
    yaw_scale: float,
    excess_time_scale: float,
    max_normalized_excess: float,
    max_linear_speed: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the single longest planted foot beyond the target stance time.

    Taking the maximum across feet makes one anchored foot the objective instead
    of allowing its cost to be traded against the behavior of the other feet.
    The excess is allowed to grow beyond one, so an indefinitely planted foot is
    increasingly expensive, while ``max_normalized_excess`` keeps the reward
    bounded against contact-sensor outliers. The command gate limits this term
    to yaw-only commands so it cannot reshape translational gaits or standing.
    """
    if excess_time_scale <= 0.0:
        raise ValueError("excess_time_scale must be greater than zero.")
    if max_normalized_excess <= 0.0:
        raise ValueError("max_normalized_excess must be greater than zero.")
    if max_linear_speed < 0.0:
        raise ValueError("max_linear_speed must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]

    command = env.command_manager.get_command(command_name)
    equivalent_speed = torch.sqrt(
        torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
    )
    target_stance_time = torch.clamp(
        max_stance_time - velocity_gain * equivalent_speed,
        min=min_stance_time,
        max=max_stance_time,
    )
    normalized_excess = torch.clamp(
        (contact_time - target_stance_time.unsqueeze(1)) / excess_time_scale,
        min=0.0,
        max=max_normalized_excess,
    )
    worst_foot_excess = torch.max(normalized_excess, dim=1).values
    command_is_pure_turn = (
        (torch.linalg.vector_norm(command[:, :2], dim=1) < max_linear_speed)
        & (torch.abs(command[:, 2]) > command_threshold)
    )
    return torch.square(worst_foot_excess) * command_is_pure_turn


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward


def velocity_conditioned_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float,
    max_air_time: float,
    velocity_gain: float,
    yaw_scale: float,
    std: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward feet that land near a velocity-conditioned target air time.

    The target decreases linearly with commanded speed and is clamped between
    ``min_air_time`` and ``max_air_time``. Yaw rate is converted to an
    approximate tangential foot speed using ``yaw_scale``. The Gaussian kernel
    keeps the per-foot landing reward bounded between zero and one.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]

    command = env.command_manager.get_command(command_name)
    equivalent_speed = torch.sqrt(
        torch.sum(torch.square(command[:, :2]), dim=1) + torch.square(yaw_scale * command[:, 2])
    )
    target_air_time = torch.clamp(
        max_air_time - velocity_gain * equivalent_speed,
        min=min_air_time,
        max=max_air_time,
    )
    normalized_error = (last_air_time - target_air_time.unsqueeze(1)) / std
    landing_reward = torch.exp(-torch.square(normalized_error)) * first_contact

    command_is_moving = torch.linalg.vector_norm(command, dim=1) > command_threshold
    return torch.sum(landing_reward, dim=1) * command_is_moving


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


class feet_lift(ManagerTermBase):
    """Reward moving feet for reaching a bounded lift above their nominal stance height.

    The lift is measured from the robot base rather than from the terrain, so this
    term does not require a height scanner or any additional policy observation.
    The nominal foot-to-base heights are captured from the configured default pose
    when the reward manager is initialized.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if cfg.params["target_height"] < 0.0:
            raise ValueError("feet_lift target_height must be non-negative.")
        if cfg.params["std"] <= 0.0:
            raise ValueError("feet_lift std must be positive.")
        if cfg.params["tanh_mult"] <= 0.0:
            raise ValueError("feet_lift tanh_mult must be positive.")

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        relative_foot_height = (
            asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        )

        # All cloned robots share the same default joint pose. Averaging over
        # environments removes tiny initialization differences while preserving
        # a separate nominal height for each foot.
        self._nominal_foot_height = relative_foot_height.mean(dim=0, keepdim=True).detach().clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        std: float,
        tanh_mult: float,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the command- and foot-speed-gated lift tracking reward."""
        asset = env.scene[asset_cfg.name]
        relative_foot_height = (
            asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        )
        foot_lift = relative_foot_height - self._nominal_foot_height

        # The Gaussian makes the reward bounded and reduces it again when a foot
        # rises above the target, discouraging unnecessarily high, wobbly steps.
        height_tracking = torch.exp(-torch.square((foot_lift - target_height) / std))
        foot_speed_xy = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
        swing_gate = torch.tanh(tanh_mult * foot_speed_xy)
        reward = torch.sum(height_tracking * swing_gate, dim=1)

        # Do not pay the robot to move its feet when it was commanded to stand.
        command = env.command_manager.get_command(command_name)
        command_is_moving = torch.linalg.vector_norm(command, dim=1) > 0.1
        return reward * command_is_moving


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
