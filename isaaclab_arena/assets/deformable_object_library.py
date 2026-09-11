# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PhysX deformable objects sourced from Isaac Lab examples."""

from __future__ import annotations

import copy
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg, PhysxSurfaceDeformableBodyMaterialCfg

from isaaclab_arena.assets.deformable_object import DeformableObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class LibraryDeformableObject(DeformableObject):
    """Base class for registered deformable objects."""

    name: str
    tags = ["object", "deformable", "physx"]
    spawner_cfg: DeformableObjectSpawnerCfg

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | PosePerEnv | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=instance_name or self.name,
            prim_path=prim_path,
            tags=self.tags,
            spawner_cfg=copy.deepcopy(self.spawner_cfg),
            initial_pose=initial_pose,
            **kwargs,
        )


@register_asset
class DeformableCube(LibraryDeformableObject):
    """PhysX deformable cube used by the DROID pick-and-place environment."""

    name = "deformable_cube"
    spawner_cfg = sim_utils.MeshCuboidCfg(
        size=(0.15, 0.04, 0.04),
        deformable_props=PhysxDeformableBodyPropertiesCfg(linear_damping=0.0),
        collision_props=[PhysxCollisionCfg(rest_offset=0.0, contact_offset=0.0025)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        physics_material=PhysxDeformableBodyMaterialCfg(
            youngs_modulus=8.0e4,
            poissons_ratio=0.25,
            density=300.0,
        ),
    )


@register_asset
class DeformableSurface(LibraryDeformableObject):
    """PhysX surface matching Isaac Lab's Franka cloth lift geometry."""

    name = "deformable_surface"
    spawner_cfg = sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        resolution=(30, 30),
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(),
    )


@register_asset
class DeformableTeddyBear(LibraryDeformableObject):
    """PhysX teddy bear from Isaac Lab's Franka lift environment."""

    name = "deformable_teddy_bear"
    spawner_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        scale=(0.01, 0.01, 0.01),
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        physics_material=PhysxDeformableBodyMaterialCfg(),
    )
