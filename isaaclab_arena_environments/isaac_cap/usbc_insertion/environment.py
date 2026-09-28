# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered Isaac Cap USB-C insertion environments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from ..registration import register_environment

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class UsbcInsertionEasyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the YAM fixed-port USB-C task."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


@dataclass
class UsbcInsertionMediumEnvironmentCfg(UsbcInsertionEasyEnvironmentCfg):
    """Configure the YAM movable-bulkhead USB-C task.

    This distinct type is required because the environment registry keys each
    factory by its config type.
    """


def _build_environment(scene_spec: Path, cfg: UsbcInsertionEasyEnvironmentCfg):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_environments.isaac_cap import register_components

    from .cameras import UsbcInsertionCameraCfg
    from .physics import configure_usbc_runtime, make_robot_spawn_cfg_addon

    register_components()
    spec = ArenaEnvGraphSpec.from_yaml(scene_spec)
    spec.embodiment.params.update(
        enable_ee_frames=True,
        use_tiled_cameras=cfg.use_tiled_cameras,
        use_instanceable_meshes=cfg.use_instanceable_meshes,
        spawn_cfg_addon=make_robot_spawn_cfg_addon(),
    )
    if cfg.enable_cameras:
        spec.embodiment.params["camera_config"] = UsbcInsertionCameraCfg()
    arena_environment = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
    assert arena_environment.env_cfg_callback is not None, "USB-C graphs must define env_cfg_override."
    arena_environment.env_cfg_callback = partial(
        configure_usbc_runtime,
        apply_graph_override=arena_environment.env_cfg_callback,
    )
    return arena_environment


@register_environment
class UsbcInsertionEasyEnvironment(ArenaEnvironmentFactory[UsbcInsertionEasyEnvironmentCfg]):
    """Build the bimanual YAM variant from its environment graph."""

    name = "vabar_contact_rich_insertion__usbc_insertion_easy"
    _legacy_argparse_cfg_type = UsbcInsertionEasyEnvironmentCfg
    scene_spec = Path(__file__).with_name("usbc_easy.yaml")

    def build(self, cfg: UsbcInsertionEasyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        return _build_environment(self.scene_spec, cfg)


@register_environment
class UsbcInsertionMediumEnvironment(ArenaEnvironmentFactory[UsbcInsertionMediumEnvironmentCfg]):
    """Build the bimanual movable-bulkhead variant from its environment graph."""

    name = "vabar_contact_rich_insertion__usbc_insertion_medium"
    _legacy_argparse_cfg_type = UsbcInsertionMediumEnvironmentCfg
    scene_spec = Path(__file__).with_name("usbc_medium.yaml")

    def build(self, cfg: UsbcInsertionMediumEnvironmentCfg) -> IsaacLabArenaEnvironment:
        return _build_environment(self.scene_spec, cfg)
