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

_NEWTON_CUBE_YOUNGS_MODULUS = 8.0e4
_NEWTON_CUBE_POISSONS_RATIO = 0.3


class LibraryDeformableObject(DeformableObject):
    """Base class for registered deformable objects."""

    registry_name: str
    """Key of this asset class in the asset registry."""

    tags: list[str]
    spawner_cfg: DeformableObjectSpawnerCfg

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | PosePerEnv | None = None,
        **kwargs: Any,
    ):
        """Create a deformable object of this asset class.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            **kwargs: Extra arguments passed to ``DeformableObject``.
        """
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            spawner_cfg=copy.deepcopy(self.spawner_cfg),
            initial_pose=initial_pose,
            **kwargs,
        )


@register_asset
class DeformableCubePhysx(LibraryDeformableObject):
    """PhysX deformable cube used by the DROID pick-and-place environment."""

    registry_name = "deformable_cube_physx"
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

    registry_name = "deformable_cube_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.MeshCuboidCfg(
        size=(0.15, 0.04, 0.04),
        edge_refinement=8.0,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        physics_material=NewtonDeformableBodyMaterialCfg(
            k_mu=_NEWTON_CUBE_YOUNGS_MODULUS / (2.0 * (1.0 + _NEWTON_CUBE_POISSONS_RATIO)),
            k_lambda=(
                _NEWTON_CUBE_YOUNGS_MODULUS
                * _NEWTON_CUBE_POISSONS_RATIO
                / ((1.0 + _NEWTON_CUBE_POISSONS_RATIO) * (1.0 - 2.0 * _NEWTON_CUBE_POISSONS_RATIO))
            ),
            k_damp=1.0e3,
            # Retain the lighter DROID cube mass so its lower-force gripper can carry it.
            density=300.0,
            particle_radius=0.0025,
        ),
    )


@register_asset
class DeformableSurfacePhysx(LibraryDeformableObject):
    """PhysX surface matching Isaac Lab's Franka cloth lift geometry."""

    registry_name = "deformable_surface_physx"
    tags = ["object", "deformable", "physx"]
    spawner_cfg = sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        edge_refinement=30,
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

    registry_name = "deformable_surface_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.MeshRectangleCfg(
        size=(0.2, 0.2),
        edge_refinement=30,
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

    registry_name = "deformable_teddy_bear_physx"
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

    registry_name = "deformable_teddy_bear_newton"
    tags = ["object", "deformable", "newton"]
    spawner_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        scale=(0.01, 0.01, 0.01),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(),
    )
