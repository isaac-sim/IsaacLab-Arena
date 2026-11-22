# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets for the domino mass-order task: look-alike dominoes of configurable mass, a runway and a bell target."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

DOMINO_SIZE_M = (0.03, 0.06, 0.16)
"""Domino thickness (X), width (Y) and height (Z) in its own frame; it topples about its Y axis."""

DOMINO_HEAVY_MASS_KG = 0.30
DOMINO_LIGHT_MASS_KG = 0.05


@register_asset
class DominoBlock(Object):
    """Upright domino; heavy and light dominoes look the same and differ only in their simulated mass."""

    registry_name = "domino_block"
    tags = ["object", "graspable", "procedural", "domino_mass_order"]

    def __init__(
        self,
        mass_kg: float = DOMINO_LIGHT_MASS_KG,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create a domino.

        Args:
            mass_kg: Simulated mass, in kilograms. The task family uses 0.30 (heavy) and 0.05 (light).
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the domino's center.
        """
        assert mass_kg > 0.0, "mass_kg must be positive."
        self.mass_kg = mass_kg
        spawner_cfg = sim_utils.CuboidCfg(
            size=DOMINO_SIZE_M,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.90, 0.80), roughness=0.5),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )


@register_asset
class DominoRunway(Object):
    """Static flat strip the dominoes are stood on."""

    registry_name = "domino_runway"
    tags = ["object", "procedural", "domino_mass_order"]

    def __init__(
        self,
        length_m: float = 0.46,
        width_m: float = 0.10,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        """Create the runway, a static box collider that runs along its Y axis.

        Args:
            length_m: Length along Y, in meters.
            width_m: Width along X, in meters.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the runway's center.
        """
        self.length_m = length_m
        self.width_m = width_m
        self.thickness_m = 0.004
        spawner_cfg = sim_utils.CuboidCfg(
            size=(width_m, length_m, self.thickness_m),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.22, 0.26), roughness=0.7),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.BASE,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )


_BRASS = (0.78, 0.60, 0.22)
_GRAPHITE = (0.16, 0.17, 0.19)


def write_domino_bell_usd(path: Path) -> None:
    """Write a kinematic bell on a post: a base disc, a post and a spherical bell whose surface a domino strikes.

    Frame: Z up, origin at the bottom center of the base.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/DominoBell").GetPrim()
    stage.SetDefaultPrim(root)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    base = UsdGeom.Cylinder.Define(stage, "/DominoBell/base")
    base.CreateRadiusAttr(0.04)
    base.CreateHeightAttr(0.01)
    base.CreateAxisAttr("Z")
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.005))
    base.CreateDisplayColorAttr([Gf.Vec3f(*_GRAPHITE)])
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())

    post = UsdGeom.Cylinder.Define(stage, "/DominoBell/post")
    post.CreateRadiusAttr(0.006)
    post.CreateHeightAttr(0.08)
    post.CreateAxisAttr("Z")
    UsdGeom.Xformable(post).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.05))
    post.CreateDisplayColorAttr([Gf.Vec3f(*_GRAPHITE)])

    bell = UsdGeom.Sphere.Define(stage, "/DominoBell/bell")
    bell.CreateRadiusAttr(0.04)
    UsdGeom.Xformable(bell).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.11))
    bell.CreateDisplayColorAttr([Gf.Vec3f(*_BRASS)])
    UsdPhysics.CollisionAPI.Apply(bell.GetPrim())

    knob = UsdGeom.Sphere.Define(stage, "/DominoBell/knob")
    knob.CreateRadiusAttr(0.008)
    UsdGeom.Xformable(knob).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.155))
    knob.CreateDisplayColorAttr([Gf.Vec3f(*_GRAPHITE)])
    stage.GetRootLayer().Save()


def get_domino_bell_usd_path() -> str:
    """Return a cached USD path for the bell, writing the file on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    path = cache_dir / f"domino_bell_{source_digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_domino_bell_usd(tmp_path)
        os.replace(tmp_path, path)
    return str(path)


@register_asset
class DominoBell(Object):
    """Kinematic bell on a post; a domino that falls against it rings it."""

    registry_name = "domino_bell"
    tags = ["object", "procedural", "domino_mass_order"]

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the bell.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the bell's base center.
        """
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            usd_path=get_domino_bell_usd_path(),
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
