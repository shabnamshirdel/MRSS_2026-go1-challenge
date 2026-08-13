# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator and action models used for Go1 actuator-response randomization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ActuatorNetMLP, ActuatorNetMLPCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class RandomizedActuatorNetMLP(ActuatorNetMLP):
    """Go1 actuator network with a correlated per-environment torque multiplier."""

    cfg: RandomizedActuatorNetMLPCfg

    def __init__(self, cfg: RandomizedActuatorNetMLPCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        lower, upper = self.cfg.motor_strength_range
        if lower <= 0.0 or upper < lower:
            raise ValueError("motor_strength_range must satisfy 0 < lower <= upper.")
        self._motor_strength = torch.ones(self._num_envs, 1, device=self._device)
        self._sample_motor_strength(slice(None))

    def _sample_motor_strength(self, env_ids: Sequence[int] | slice) -> None:
        lower, upper = self.cfg.motor_strength_range
        samples = torch.empty_like(self._motor_strength[env_ids]).uniform_(lower, upper)
        self._motor_strength[env_ids] = samples

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._sample_motor_strength(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        self.computed_effort *= self._motor_strength
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class RandomizedActuatorNetMLPCfg(ActuatorNetMLPCfg):
    """Configuration for per-environment strength randomization of an actuator network."""

    class_type: type = RandomizedActuatorNetMLP
    motor_strength_range: tuple[float, float] = (1.0, 1.0)


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action with episode-randomized fractional response smoothing."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        if (
            cfg.min_delay_fraction < 0.0
            or cfg.max_delay_fraction < cfg.min_delay_fraction
            or cfg.max_delay_fraction > 1.0
        ):
            raise ValueError(
                "Action delay fraction must satisfy "
                "0 <= min_delay_fraction <= max_delay_fraction <= 1."
            )
        super().__init__(cfg, env)
        if isinstance(self._offset, torch.Tensor):
            self._previous_processed_actions = self._offset.clone()
        else:
            self._previous_processed_actions = torch.full_like(self._processed_actions, self._offset)
        self._delay_fraction = torch.zeros(self.num_envs, 1, device=self.device)
        self._sample_delay(slice(None))

    def _sample_delay(self, env_ids: Sequence[int] | slice) -> None:
        samples = torch.empty_like(self._delay_fraction[env_ids]).uniform_(
            self.cfg.min_delay_fraction, self.cfg.max_delay_fraction
        )
        self._delay_fraction[env_ids] = samples

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        current_processed_actions = self._processed_actions.clone()
        self._processed_actions = (
            self._delay_fraction * self._previous_processed_actions
            + (1.0 - self._delay_fraction) * current_processed_actions
        )
        self._previous_processed_actions[:] = current_processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        if isinstance(self._offset, torch.Tensor):
            self._previous_processed_actions[resolved_env_ids] = self._offset[resolved_env_ids]
        else:
            self._previous_processed_actions[resolved_env_ids] = self._offset
        self._sample_delay(resolved_env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for fractional joint-position response smoothing."""

    class_type: type = DelayedJointPositionAction
    min_delay_fraction: float = 0.0
    max_delay_fraction: float = 0.0
