# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the fused block tower task: the procedural tower, its free-pair variation and the side pad."""

from __future__ import annotations

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.fused_block_tower_usd import ALL_FREE_PAIRS, get_fused_block_tower_usd_path, pair_key
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg


@configclass
class FreePairVariationCfg(VariationBaseCfg):
    """Configuration for FreePairVariation."""

    free_pairs: list[list[str]] = field(default_factory=lambda: [list(pair) for pair in ALL_FREE_PAIRS])
    """Free pairs to draw from, each two dot-marked block names in different layers, e.g. [[L1_a,L3_c]]."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform choice over ``free_pairs``, seeded by the build seed."""


class FreePairVariation(BuildTimeVariationBase):
    """Pick which two dot-marked blocks of a fused block tower are free, once per environment build.

    Every environment of a build shares the tower USD, so they share the free pair.

    Args:
        tower: The tower whose USD the variation selects.
        cfg: Variation settings. By default the variation is off and the tower keeps its ``free_pair`` parameter.
        name: Name of the variation on the tower.
    """

    cfg: FreePairVariationCfg

    def __init__(self, tower: FusedBlockTower, cfg: FreePairVariationCfg | None = None, name: str = "free_pair"):
        super().__init__(cfg=cfg if cfg is not None else FreePairVariationCfg(), name=name)
        self._tower = tower

    def _realize_at_build_time(self) -> None:
        assert self.cfg.free_pairs, "free_pair variation is enabled but free_pairs is empty."
        free_pairs = [list(pair_key(pair)) for pair in self.cfg.free_pairs]
        free_pair = self.sampler.sample(num_samples=1, choices=free_pairs)[0]
        self._tower.set_free_pair(free_pair)


@register_asset
class FusedBlockTower(Object):
    """Five-layer block tower in which every dot-marked block except a hidden free pair is fused to the layer above.

    The tower is a static (``BASE``) asset whose USD holds one rigid body per block; refer to the blocks with rigid
    object references (prim paths ``L<layer>_<a|b|c>``). See fused_block_tower_usd.py for the layout. The free pair is
    set by the ``free_pair`` parameter, or per build by the tower's ``free_pair`` variation.
    """

    registry_name = "fused_block_tower"
    tags = ["object", "procedural", "fused_block_tower"]

    def __init__(
        self,
        free_pair: list[str] | tuple[str, str] = ("L1_a", "L3_c"),
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """Create the tower.

        Args:
            free_pair: The two dot-marked blocks without a fuse joint, in different layers, for example
                ``["L1_a", "L3_c"]``.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the tower's bottom center.
            **kwargs: Arguments passed to ``Object``.
        """
        self.free_pair = pair_key(free_pair)
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.BASE,
            usd_path=get_fused_block_tower_usd_path(self.free_pair),
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            },
            **kwargs,
        )
        self.add_variation(FreePairVariation(self))

    def set_free_pair(self, free_pair: list[str] | tuple[str, str]) -> None:
        """Spawn the tower variant whose free pair is ``free_pair``."""
        self.free_pair = pair_key(free_pair)
        self.usd_path = get_fused_block_tower_usd_path(self.free_pair)
        self.object_cfg.spawn.usd_path = self.usd_path
        print(f"[fused_block_tower] '{self.name}' free pair: {list(self.free_pair)}")


@register_asset
class FusedTowerSidePad(Object):
    """Flat, high-friction square pad that removed tower blocks are placed on."""

    registry_name = "fused_tower_side_pad"
    tags = ["object", "procedural", "fused_block_tower"]

    def __init__(
        self,
        size_m: float = 0.13,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create the pad.

        Args:
            size_m: Edge length of the square pad, in meters.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the pad's center.
        """
        self.size_m = size_m
        self.thickness_m = 0.008
        spawner_cfg = sim_utils.CuboidCfg(
            size=(size_m, size_m, self.thickness_m),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.36, 0.30), roughness=0.8),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
