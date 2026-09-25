# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe factories with camera, placement, and physics adaptations."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from ...registration import register_environment


def _apply_syringe_graph_config(env_cfg, graph_callback):
    """Apply the remaining Python physics settings and the graph's configuration."""
    # TODO(alexmillane) [isaaclab-multiccd-config-missing-feature]: Move this to YAML
    # once Isaac Lab exposes enable_multiccd in MJWarpSolverCfg.
    env_cfg.sim.physics.solver_cfg.enable_multiccd = True
    return graph_callback(env_cfg)


@dataclass
class SyringeSortEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the syringe environment and an optional episode timeout."""

    enable_cameras: bool = False
    episode_length_s: float | None = None


class SyringeBase(ArenaEnvironmentFactory[SyringeSortEnvironmentCfg]):
    """Pick a syringe from its tray and release it into the sharps container."""

    yaml_file: str

    def build(self, cfg: SyringeSortEnvironmentCfg):
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena_environments.isaac_cap import register_components

        from .cameras import configure_syringe_cameras

        register_components()
        spec = ArenaEnvGraphSpec.from_yaml(str(Path(__file__).with_name(self.yaml_file)))
        arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        # NOTE(alexmillane, 2028.09.17) [clutter-placement-missing-feature]:
        # Move to clutter-based placement when that feature is enabled.

        # TODO(alexmillane) [berkley-cap-align-embodiments]: Remove these per-task custom
        # embodiment configurations once the upstream repo has done it.
        configure_syringe_cameras(arena_env.embodiment.camera_config)
        gripper = arena_env.embodiment.scene_config.robot.actuators["robotiq_driver"]
        gripper.stiffness, gripper.damping = 20.0, 1.0
        arena_env.embodiment.action_config.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_driver_joint"],
            preserve_order=True,
            use_default_offset=False,
            scale=0.8,
            offset=0.0,
        )
        if cfg.episode_length_s is not None:
            assert cfg.episode_length_s > 0
            arena_env.task.episode_length_s = cfg.episode_length_s
        arena_env.env_cfg_callback = partial(_apply_syringe_graph_config, graph_callback=arena_env.env_cfg_callback)
        return arena_env


@register_environment(cfg_type=SyringeSortEnvironmentCfg)
class SyringeSingleEnvironment(SyringeBase):
    """Dispose of one syringe from a fixed layout."""

    name = "syringe_single_newton"
    yaml_file = "syringe_single.yaml"
    _legacy_argparse_cfg_type = SyringeSortEnvironmentCfg


@dataclass
class SyringeBothEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Configure the randomized two-syringe benchmark."""


@register_environment(cfg_type=SyringeBothEnvironmentCfg)
class SyringeBothEnvironment(SyringeBase):
    """Dispose of both the red-cap and white-cap syringes."""

    name = "syringe_both_newton"
    yaml_file = "syringe_both.yaml"
    _legacy_argparse_cfg_type = SyringeBothEnvironmentCfg


@dataclass
class SyringeClutteredEnvironmentCfg(SyringeBothEnvironmentCfg):
    """Configure the randomized four-syringe benchmark."""


@register_environment(cfg_type=SyringeClutteredEnvironmentCfg)
class SyringeClutteredEnvironment(SyringeBothEnvironment):
    """Dispose of all four syringes from the cluttered tray."""

    name = "syringe_cluttered_newton"
    yaml_file = "syringe_cluttered.yaml"
    _legacy_argparse_cfg_type = SyringeClutteredEnvironmentCfg
