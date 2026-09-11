# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered Isaac Cap bimanual YAM cable-routing environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class CableRoutingMediumEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure Cap's medium two-peg cable route."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


@dataclass
class CableRoutingEasyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure Cap's easy straight-cable route."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


def _build_environment(
    factory: ArenaEnvironmentFactory,
    cfg: CableRoutingMediumEnvironmentCfg | CableRoutingEasyEnvironmentCfg,
    variant_name: str,
) -> IsaacLabArenaEnvironment:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    from ..embodiments.cable_routing import IndustrialBimanualYamEmbodiment
    from .physics import configure_cable_routing_physics
    from .scene import (
        BOARD_TOP_Z,
        EASY_VARIANT,
        LEFT_YAM_POSITION,
        MEDIUM_VARIANT,
        RIGHT_YAM_POSITION,
        TABLE_CENTER_X,
        YAM_INSTANCEABLE_USD_PATH,
        YAM_USD_PATH,
        build_cable_routing_scene,
    )
    from .task import CableRoutingTask

    assert variant_name in ("easy", "medium"), f"Unsupported cable-routing variant {variant_name!r}."
    variant = EASY_VARIANT if variant_name == "easy" else MEDIUM_VARIANT
    built_scene = build_cable_routing_scene(factory.asset_registry, factory.hdr_registry, variant)
    embodiment = IndustrialBimanualYamEmbodiment(
        robot_usd_path=YAM_USD_PATH,
        instanceable_robot_usd_path=YAM_INSTANCEABLE_USD_PATH,
        left_mount_position=LEFT_YAM_POSITION,
        right_mount_position=RIGHT_YAM_POSITION,
        enable_cameras=cfg.enable_cameras,
        use_tiled_cameras=cfg.use_tiled_cameras,
        use_instanceable_meshes=cfg.use_instanceable_meshes,
    )
    task = CableRoutingTask(
        cable=built_scene.cable,
        pegs=built_scene.pegs,
        route_peg_indices=variant.route_peg_indices,
        route_directions=variant.route_directions,
        task_description=variant.task_description,
        viewer_lookat=(TABLE_CENTER_X, 0.0, BOARD_TOP_Z),
    )
    return IsaacLabArenaEnvironment(
        name=f"vabar_cable_routing__{variant.name}",
        embodiment=embodiment,
        scene=built_scene.scene,
        task=task,
        env_cfg_callback=configure_cable_routing_physics,
    )


class CableRoutingMediumEnvironment(ArenaEnvironmentFactory[CableRoutingMediumEnvironmentCfg]):
    """Build Cap's medium cable-routing environment on native Arena APIs."""

    name = "vabar_cable_routing__medium"
    _legacy_argparse_cfg_type = CableRoutingMediumEnvironmentCfg

    def build(self, cfg: CableRoutingMediumEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Compose the medium cable route."""
        return _build_environment(self, cfg, "medium")


class CableRoutingEasyEnvironment(ArenaEnvironmentFactory[CableRoutingEasyEnvironmentCfg]):
    """Build Cap's easy cable-routing environment on native Arena APIs."""

    name = "vabar_cable_routing__easy"
    _legacy_argparse_cfg_type = CableRoutingEasyEnvironmentCfg

    def build(self, cfg: CableRoutingEasyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Compose the easy cable route."""
        return _build_environment(self, cfg, "easy")
