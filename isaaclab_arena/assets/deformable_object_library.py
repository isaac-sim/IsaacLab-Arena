# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-specific deformable objects sourced from Isaac Lab examples."""

from __future__ import annotations

import copy
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg, PhysxSurfaceDeformableBodyMaterialCfg

from isaaclab_arena.assets.deformable_object import DeformableObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class LibraryDeformableObject(DeformableObject):
    """Base class for registered deformable objects."""

    name: str
    tags: list[str]
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
class DeformableCubePhysx(LibraryDeformableObject):
    """PhysX deformable cube used by the DROID pick-and-place environment."""

    name = "deformable_cube_physx"
    tags = ["object", "deformable", "physx"]
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
class DeformableCubeNewton(LibraryDeformableObject):
    """Newton equivalent of the DROID pick-and-place deformable cube."""

    name = "deformable_cube_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.MeshCuboidCfg(
        size=(0.15, 0.04, 0.04),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        physics_material=NewtonDeformableBodyMaterialCfg(
            # Equivalent Lame parameters for the PhysX cube's E=8.0e4 Pa and nu=0.25.
            k_mu=3.2e4,
            k_lambda=3.2e4,
            k_damp=0.0,
            density=300.0,
            particle_radius=0.0025,
        ),
    )


@register_asset
class DeformableSurfacePhysx(LibraryDeformableObject):
    """PhysX surface matching Isaac Lab's Franka cloth lift geometry."""

    name = "deformable_surface_physx"
    tags = ["object", "deformable", "physx"]
    spawner_cfg = sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        resolution=(30, 30),
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        collision_props=[PhysxCollisionCfg(rest_offset=0.002, contact_offset=0.01)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
            density=1000.0,
            surface_thickness=0.001,
            poissons_ratio=0.25,
            youngs_modulus=1e6,
            surface_bend_stiffness=1e6,
            elasticity_damping=1e-1,
            bend_damping=1e-1,
            static_friction=10.0,
            dynamic_friction=10.0,
        ),
    )


@register_asset
class DeformableSurfaceNewton(LibraryDeformableObject):
    """Newton surface using Isaac Lab's Franka cloth lift material."""

    name = "deformable_surface_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        resolution=(30, 30),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
        physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
            density=1.0,
            particle_radius=0.002,
            tri_ke=5e2,
            tri_ka=5e2,
            tri_kd=1e-3,
            edge_ke=0.5,
            edge_kd=1e-3,
        ),
    )


@register_asset
class DeformableTeddyBearPhysx(LibraryDeformableObject):
    """PhysX teddy bear from Isaac Lab's Franka lift environment."""

    name = "deformable_teddy_bear_physx"
    tags = ["object", "deformable", "physx"]
    spawner_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        scale=(0.01, 0.01, 0.01),
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        physics_material=PhysxDeformableBodyMaterialCfg(),
    )


@register_asset
class DeformableTeddyBearNewton(LibraryDeformableObject):
    """Newton teddy bear matching Isaac Lab's deformables demo."""

    name = "deformable_teddy_bear_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        scale=(0.01, 0.01, 0.01),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(),
    )
