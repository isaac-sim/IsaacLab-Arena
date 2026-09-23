# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply validated overrides to an Isaac Lab Arena environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_newton.physics import NewtonCfg

from isaaclab_arena.hydra.config_override import apply_config_override

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def apply_env_cfg_override(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    override: dict[str, Any],
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply a validated graph ``env_cfg_override`` in place."""
    apply_config_override(
        env_cfg,
        override,
        override_name="env_cfg_override",
        path="env",
    )
    # NewtonCfg derives its manager class from the initial solver in __post_init__, so resynchronize it
    # when an override replaces solver_cfg.
    physics_cfg = env_cfg.sim.physics
    if isinstance(physics_cfg, NewtonCfg):
        expected_class = physics_cfg.solver_cfg.class_type
        if physics_cfg.class_type is not expected_class:
            physics_cfg.class_type = expected_class
    return env_cfg
