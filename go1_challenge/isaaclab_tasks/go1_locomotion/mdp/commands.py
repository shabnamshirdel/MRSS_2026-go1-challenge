"""Velocity-command generators specific to the Go1 locomotion task."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass


class PureTurnUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity commands with an explicit fraction of in-place turns."""

    cfg: "PureTurnUniformVelocityCommandCfg"

    def __init__(self, cfg: "PureTurnUniformVelocityCommandCfg", env):
        if not 0.0 <= cfg.rel_pure_turn_envs <= 1.0:
            raise ValueError("rel_pure_turn_envs must be in the range [0, 1].")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pure_turn_mask = (
            torch.rand(env_ids_tensor.numel(), device=self.device) < self.cfg.rel_pure_turn_envs
        )
        pure_turn_env_ids = env_ids_tensor[pure_turn_mask]
        self.vel_command_b[pure_turn_env_ids, :2] = 0.0


@configclass
class PureTurnUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for velocity commands containing in-place-turn samples."""

    class_type: type[PureTurnUniformVelocityCommand] | str = PureTurnUniformVelocityCommand
    rel_pure_turn_envs: float = 0.25

