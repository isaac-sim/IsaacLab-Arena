# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered bimanual YAM cable-routing environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class YamCableRoutingEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the bimanual YAM cable-routing environment."""

    episode_length_s: float = 3600.0
    """Episode timeout in seconds."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be positive."


@register_environment
class YamCableRoutingEnvironment(ArenaEnvironmentFactory[YamCableRoutingEnvironmentCfg]):
    """Compose the YAM embodiment, cable-routing scene, task, and coupled Newton solver."""

    name = "yam_cable_routing"
    _legacy_argparse_cfg_type = YamCableRoutingEnvironmentCfg

    def build(self, cfg: YamCableRoutingEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the production YAM cable-routing environment."""
        from isaaclab_arena.embodiments.yam.yam import BimanualYamEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.tasks.cable_routing_task import CableRoutingTask
        from isaaclab_arena_environments.yam_cable_routing.physics import configure_yam_cable_routing_physics
        from isaaclab_arena_environments.yam_cable_routing.scene import (
            BOARD_TOP_Z,
            CABLE_RADIUS,
            LEFT_YAM_POSITION,
            PEG_HEIGHT,
            RIGHT_YAM_POSITION,
            TABLE_CENTER_X,
            build_yam_cable_routing_scene,
        )

        scene_assets = build_yam_cable_routing_scene()
        embodiment_type = self.asset_registry.get_asset_by_name("yam_bimanual")
        assert embodiment_type is BimanualYamEmbodiment, "yam_bimanual resolved to an unexpected asset type."
        embodiment = embodiment_type(
            enable_cameras=cfg.enable_cameras,
            left_position=LEFT_YAM_POSITION,
            right_position=RIGHT_YAM_POSITION,
        )
        task = CableRoutingTask(
            cable=scene_assets.cable,
            pegs=scene_assets.pegs,
            route_directions=(-1.0, 1.0),
            episode_length_s=cfg.episode_length_s,
            axial_cutoff=0.5 * PEG_HEIGHT + CABLE_RADIUS,
            viewer_lookat=(TABLE_CENTER_X, 0.0, BOARD_TOP_Z),
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene_assets.scene,
            task=task,
            env_cfg_callback=configure_yam_cable_routing_physics,
            # The CouplerProxy configuration is part of the task contract. A
            # global PhysX or stock Newton preset would silently replace it.
            supported_physics_presets=(),
        )
