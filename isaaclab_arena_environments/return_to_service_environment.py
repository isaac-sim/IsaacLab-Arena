# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A DROID workcell for diagnosing, servicing, and re-kitting returned vacuums."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ReturnToServiceEnvironmentCfg(ArenaEnvironmentCfg):
    """Select the asset bundle, DROID controls, and return conditions."""

    asset_root: str | None = None
    """Blender-generated bundle; defaults to the user's Arena asset cache."""

    embodiment: str = "droid_differential_ik"
    """Existing Arena DROID embodiment, preserving its action and observation contracts."""

    scenarios: list[str] = field(default_factory=lambda: ["combined"])
    """Fault conditions assigned cyclically across parallel environments."""

    episode_length_s: float = 600.0
    """Simulated-time budget for the complete servicing and packing episode."""

    table_height_m: float = 0.78
    """Workbench surface and robot mounting height."""

    gripper_stiffness: float = 4.0
    """Existing gripper driver's stiffness in N m/rad for this workcell."""

    gripper_damping: float = 1.0
    """Existing gripper driver's damping in N m s/rad for this workcell."""


def _configure_service_physics(env_cfg, *, gripper_stiffness: float, gripper_damping: float):
    """Configure workcell contacts and the existing single-driver gripper before construction."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_physx.physics import PhysxCfg

    assert isinstance(env_cfg.sim.physics, PhysxCfg), "Return-to-service currently requires PhysX."
    assert math.isfinite(gripper_stiffness) and gripper_stiffness > 0, "Gripper stiffness must be positive and finite."
    assert math.isfinite(gripper_damping) and gripper_damping > 0, "Gripper damping must be positive and finite."
    gripper = env_cfg.scene.robot.actuators["gripper"]
    assert isinstance(gripper, ImplicitActuatorCfg) and gripper.joint_names_expr == [
        "finger_joint"
    ], "Workcell tuning requires DROID's existing single-driver implicit gripper."
    gripper.stiffness = gripper_stiffness
    gripper.damping = gripper_damping
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.decimation = 4
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg


@register_environment
class ReturnToServiceEnvironment(ArenaEnvironmentFactory[ReturnToServiceEnvironmentCfg]):
    """Compose the service scene, existing DROID embodiment, and diagnostic task."""

    name = "return_to_service"
    _legacy_argparse_cfg_type = ReturnToServiceEnvironmentCfg

    def build(self, cfg: ReturnToServiceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.return_to_service.scene import build_service_scene
        from isaaclab_arena_environments.return_to_service.task import ReturnToServiceTask

        assert cfg.embodiment.startswith("droid_"), "This example requires an existing DROID embodiment."
        workcell = build_service_scene(cfg.asset_root, table_height_m=cfg.table_height_m)
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            initial_pose=Pose(position_xyz=(0.0, 0.0, cfg.table_height_m)),
            stand_height_m=cfg.table_height_m,
            stand_footprint_xy_m=(0.20, 0.20),
        )
        task = ReturnToServiceTask(workcell, cfg.scenarios, episode_length_s=cfg.episode_length_s)
        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=workcell.scene,
            embodiment=embodiment,
            task=task,
            env_cfg_callback=partial(
                _configure_service_physics,
                gripper_stiffness=cfg.gripper_stiffness,
                gripper_damping=cfg.gripper_damping,
            ),
        )
