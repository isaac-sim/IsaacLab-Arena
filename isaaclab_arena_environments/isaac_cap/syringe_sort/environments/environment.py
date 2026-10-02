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
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena_environments.isaac_cap import register_components

        from .cameras import SyringeCameraCfg

        register_components()
        spec = ArenaEnvGraphSpec.from_yaml(str(Path(__file__).with_name(self.yaml_file)))
        arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        cameras = SyringeCameraCfg()
        cameras.use_tiled_camera = arena_env.embodiment.camera_config.use_tiled_camera
        arena_env.embodiment.camera_config = cameras
        if cfg.episode_length_s is not None:
            assert cfg.episode_length_s > 0
            arena_env.task.episode_length_s = cfg.episode_length_s
        arena_env.env_cfg_callback = partial(_apply_syringe_graph_config, graph_callback=arena_env.env_cfg_callback)
        return arena_env


@register_environment(cfg_type=SyringeSortEnvironmentCfg)
class SyringeSingleEnvironment(SyringeBase):
    """Dispose of one red-cap syringe from a randomized tray."""

    name = "syringe_single_newton"
    yaml_file = "syringe_single.yaml"
    _legacy_argparse_cfg_type = SyringeSortEnvironmentCfg


@dataclass
class SyringeBothEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Configure the randomized two-syringe benchmark."""


@register_environment(cfg_type=SyringeBothEnvironmentCfg)
class SyringeBothEnvironment(SyringeBase):
    """Dispose of both red-cap syringes."""

    name = "syringe_both_newton"
    yaml_file = "syringe_both.yaml"
    _legacy_argparse_cfg_type = SyringeBothEnvironmentCfg


@dataclass
class SyringeDesignatedEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Configure the red-cap syringe benchmark with a blank distractor."""


@register_environment(cfg_type=SyringeDesignatedEnvironmentCfg)
class SyringeDesignatedEnvironment(SyringeBase):
    """Dispose of the red-cap syringe beside an unscored blank syringe."""

    name = "syringe_designated_newton"
    yaml_file = "syringe_designated.yaml"
    _legacy_argparse_cfg_type = SyringeDesignatedEnvironmentCfg


@dataclass
class SyringeClutteredEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Configure the randomized six-syringe benchmark."""


@register_environment(cfg_type=SyringeClutteredEnvironmentCfg)
class SyringeClutteredEnvironment(SyringeBase):
    """Dispose of all six syringes from the cluttered tray."""

    name = "syringe_cluttered_newton"
    yaml_file = "syringe_cluttered.yaml"
    _legacy_argparse_cfg_type = SyringeClutteredEnvironmentCfg
