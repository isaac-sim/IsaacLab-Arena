# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current-CAP cable environments backed by Arena Cable."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from ..registration import register_environment

if TYPE_CHECKING:
    from .scene import CableRoutingVariant


@dataclass
class CableRoutingMediumEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure one seeded current-CAP Medium terminated-weave layout."""

    layout_seed: int = 10000
    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False
    cable_camera_width: int = 1280


@dataclass
class CableRoutingEasyEnvironmentCfg(CableRoutingMediumEnvironmentCfg):
    """Configure one seeded current-CAP Easy terminated-weave layout."""

    layout_seed: int = 10009


@register_environment(cfg_type=CableRoutingMediumEnvironmentCfg)
class CableRoutingMediumEnvironment(ArenaEnvironmentFactory[CableRoutingMediumEnvironmentCfg]):
    """Build a current four-guide Medium layout with an Arena Cable."""

    name = "vabar_cable_routing_v2__medium"
    _legacy_argparse_cfg_type = CableRoutingMediumEnvironmentCfg

    def _variant_for_cfg(self, cfg: CableRoutingMediumEnvironmentCfg) -> CableRoutingVariant:
        from .scene import medium_variant

        return medium_variant(cfg.layout_seed)

    def build(self, cfg: CableRoutingMediumEnvironmentCfg):
        """Compose the current CAP layout from staging-bucket assets."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.utils.physics_backend import PhysicsBackend

        from .physics import configure_cable_routing_physics
        from .scene import build_cable_routing_scene
        from .task import CableRoutingTaskV2
        from .yam_i2rt import CableRoutingYamI2rtEmbodiment

        variant = self._variant_for_cfg(cfg)
        built_scene = build_cable_routing_scene(variant)
        embodiment = CableRoutingYamI2rtEmbodiment(
            model_position=(-0.335, 0.0, 0.767),
            enable_cameras=cfg.enable_cameras,
            use_tiled_cameras=cfg.use_tiled_cameras,
            cable_camera_width=cfg.cable_camera_width,
            medium=self.name == CableRoutingMediumEnvironment.name,
        )
        task = CableRoutingTaskV2(
            cable=built_scene.cable,
            pegs=built_scene.pegs,
            port=built_scene.port,
            variant=variant,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=built_scene.scene,
            task=task,
            env_cfg_callback=partial(
                configure_cable_routing_physics,
                physics=variant.physics,
                pin_start=variant.pin_start,
            ),
            default_physics_backend=PhysicsBackend.NEWTON,
        )


@register_environment(cfg_type=CableRoutingEasyEnvironmentCfg)
class CableRoutingEasyEnvironment(CableRoutingMediumEnvironment):
    """Build a current two-guide Easy layout with an Arena Cable."""

    name = "vabar_cable_routing_v2__easy"
    _legacy_argparse_cfg_type = CableRoutingEasyEnvironmentCfg

    def _variant_for_cfg(self, cfg: CableRoutingMediumEnvironmentCfg) -> CableRoutingVariant:
        from .scene import easy_variant

        assert isinstance(cfg, CableRoutingEasyEnvironmentCfg)
        return easy_variant(cfg.layout_seed)


__all__ = [
    "CableRoutingEasyEnvironment",
    "CableRoutingEasyEnvironmentCfg",
    "CableRoutingMediumEnvironment",
    "CableRoutingMediumEnvironmentCfg",
]
