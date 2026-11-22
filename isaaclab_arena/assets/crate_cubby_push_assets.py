# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets of the crate-into-cubby push task: a sealed crate with a hidden center of mass, a snug cubby and a rubber mat.

The crate is a procedural rigid body whose center of mass is authored explicitly with UsdPhysics.MassAPI, so its
offset is invisible from outside. Pushed at height h on its rear face, the crate tips about its leading bottom edge
when mu * h exceeds the horizontal distance from its center of mass to that edge, and it yaws when the push line misses
its center of mass sideways. The six registered crate variants cover the original task's center-of-mass states.

Crate frame: origin at the center of the runner soles, +X toward the leading (front) face, +Y to the crate's left, +Z
up. Cubby frame: origin on the floor at the center of the entrance plane, +X into the cubby, +Z up.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


def _cached_usd_path(prefix: str, key: str, writer) -> str:
    """Return a cached USD path for a content key, writing it atomically on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache_dir / f"{prefix}_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        writer(tmp_path)
        os.replace(tmp_path, path)
    return str(path)


def _new_stage(path: Path, root_name: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def _define_physics_material(stage, path: str, static_friction: float, dynamic_friction: float):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(static_friction)
    material_api.CreateDynamicFrictionAttr(dynamic_friction)
    material_api.CreateRestitutionAttr(0.0)
    return material


def _add_box(stage, path: str, min_xyz, max_xyz, color, material=None) -> None:
    """Add an axis-aligned box spanning min_xyz to max_xyz in its parent frame; it collides when given a material."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2 for lo, hi in zip(min_xyz, max_xyz))))
    xform.AddScaleOp().Set(Gf.Vec3f(*(hi - lo for lo, hi in zip(min_xyz, max_xyz))))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if material is not None:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )


@dataclass(frozen=True)
class SealedCrateGeometry:
    """Dimensions (m), mass (kg), center of mass (m) and friction of the procedural sealed crate."""

    depth_m: float = 0.14
    """Size along the push direction (+X)."""

    width_m: float = 0.104
    height_m: float = 0.135
    """Total height, runners included."""

    runner_height_m: float = 0.012
    runner_width_m: float = 0.016
    mass_kg: float = 0.38
    com_forward_m: float = 0.04
    """Center-of-mass offset toward the leading face; negative is toward the rear face."""

    com_left_m: float = 0.0
    """Center-of-mass offset toward the crate's left (+Y)."""

    com_height_m: float = 0.08
    """Center-of-mass height above the runner soles."""

    runner_static_friction: float = 0.7
    runner_dynamic_friction: float = 0.6
    body_static_friction: float = 0.35
    body_dynamic_friction: float = 0.3


_CRATE_WOOD = (0.64, 0.47, 0.28)
_CRATE_TRIM = (0.30, 0.20, 0.12)
_RUBBER = (0.10, 0.10, 0.11)


def write_sealed_crate_usd(path: Path, geometry: SealedCrateGeometry) -> None:
    """Write the sealed crate as one rigid body with an explicit mass, center of mass and inertia.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions, mass, center of mass and friction of the crate.
    """
    from pxr import Gf, UsdPhysics

    g = geometry
    stage, root = _new_stage(path, "SealedCrate")
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(g.mass_kg)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(g.com_forward_m, g.com_left_m, g.com_height_m))
    # Solid-box inertia about the center of mass; the offset load is treated as part of the uniform box.
    body_height = g.height_m
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            g.mass_kg * (g.width_m**2 + body_height**2) / 12.0,
            g.mass_kg * (g.depth_m**2 + body_height**2) / 12.0,
            g.mass_kg * (g.depth_m**2 + g.width_m**2) / 12.0,
        )
    )
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

    rubber = _define_physics_material(
        stage, "/SealedCrate/PhysicsMaterials/runner_rubber", g.runner_static_friction, g.runner_dynamic_friction
    )
    wood = _define_physics_material(
        stage, "/SealedCrate/PhysicsMaterials/crate_wood", g.body_static_friction, g.body_dynamic_friction
    )
    half_depth, half_width = g.depth_m / 2, g.width_m / 2
    _add_box(
        stage,
        "/SealedCrate/body",
        (-half_depth, -half_width, g.runner_height_m),
        (half_depth, half_width, g.height_m),
        _CRATE_WOOD,
        wood,
    )
    runner_center_y = half_width - g.runner_width_m / 2
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        _add_box(
            stage,
            f"/SealedCrate/runner_{side_name}",
            (-half_depth, side * runner_center_y - g.runner_width_m / 2, 0.0),
            (half_depth, side * runner_center_y + g.runner_width_m / 2, g.runner_height_m + 0.001),
            _RUBBER,
            rubber,
        )
    # Visual trim only: a lid seam and two bands, flush within 1 mm of the faces.
    lid_z = g.height_m - 0.012
    _add_box(
        stage,
        "/SealedCrate/lid_seam",
        (-half_depth - 0.0005, -half_width - 0.0005, lid_z - 0.002),
        (half_depth + 0.0005, half_width + 0.0005, lid_z + 0.002),
        _CRATE_TRIM,
    )
    for band_index, band_z in enumerate((g.runner_height_m + 0.03, g.height_m - 0.04)):
        _add_box(
            stage,
            f"/SealedCrate/band_{band_index}",
            (-half_depth - 0.0008, -half_width - 0.0008, band_z - 0.006),
            (half_depth + 0.0008, half_width + 0.0008, band_z + 0.006),
            _CRATE_TRIM,
        )
    stage.GetRootLayer().Save()


def get_sealed_crate_usd_path(geometry: SealedCrateGeometry) -> str:
    """Return a cached USD path for a crate variant, writing the file on first use."""
    return _cached_usd_path("sealed_crate", repr(geometry), lambda path: write_sealed_crate_usd(path, geometry))


@register_asset
class SealedCrate(Object):
    """Sealed crate on rubber runners whose center of mass is offset by a hidden amount.

    The USD is generated per center-of-mass setting and cached by content hash. The registered variants
    ``sealed_crate_<front|rear>_heavy_<left|center|right>`` fix the offset so an asset selection can choose it per run.
    """

    registry_name = "sealed_crate"
    tags = ["object", "pushable", "procedural", "crate_cubby_push"]
    com_forward_m: float = 0.04
    com_left_m: float = 0.0

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        com_forward_m: float | None = None,
        com_left_m: float | None = None,
        mass_kg: float = 0.38,
    ):
        """Create the crate.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            com_forward_m: Center-of-mass offset toward the leading face, in meters. Defaults to the class value.
            com_left_m: Center-of-mass offset toward the crate's left, in meters. Defaults to the class value.
            mass_kg: Crate mass in kilograms.
        """
        self.geometry = SealedCrateGeometry(
            mass_kg=mass_kg,
            com_forward_m=self.com_forward_m if com_forward_m is None else com_forward_m,
            com_left_m=self.com_left_m if com_left_m is None else com_left_m,
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            usd_path=get_sealed_crate_usd_path(self.geometry),
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            },
        )


def _register_crate_variant(fore_aft: str, com_forward_m: float, side: str, com_left_m: float) -> None:
    """Register a SealedCrate subclass with a fixed center-of-mass offset."""
    registry_name = f"sealed_crate_{fore_aft}_heavy_{side}"
    variant = type(
        f"SealedCrate{fore_aft.capitalize()}Heavy{side.capitalize()}",
        (SealedCrate,),
        {
            "__doc__": (
                f"Sealed crate with its center of mass {abs(com_forward_m) * 100:.0f} cm toward its {fore_aft} face"
                f" and {abs(com_left_m) * 100:.0f} cm to the {side}."
            ),
            "__module__": __name__,
            "registry_name": registry_name,
            "com_forward_m": com_forward_m,
            "com_left_m": com_left_m,
        },
    )
    register_asset(variant)


for _fore_aft, _com_forward_m in (("front", 0.04), ("rear", -0.04)):
    for _side, _com_left_m in (("left", 0.03), ("center", 0.0), ("right", -0.03)):
        _register_crate_variant(_fore_aft, _com_forward_m, _side, _com_left_m)


@dataclass(frozen=True)
class CubbyShelfGeometry:
    """Interior dimensions (m) and wall friction of the procedural cubby under a shelf board."""

    interior_width_m: float = 0.124
    interior_depth_m: float = 0.17
    interior_height_m: float = 0.155
    wall_thickness_m: float = 0.015
    top_thickness_m: float = 0.018
    top_setback_m: float = 0.04
    """Distance the shelf board's front edge sits behind the entrance plane, leaving room for a pushing hand."""

    top_overhang_m: float = 0.03
    static_friction: float = 0.3
    dynamic_friction: float = 0.25


DEFAULT_CUBBY_GEOMETRY = CubbyShelfGeometry()
_CUBBY_WALL = (0.86, 0.84, 0.80)
_CUBBY_TOP = (0.45, 0.32, 0.20)


def write_cubby_shelf_usd(path: Path, geometry: CubbyShelfGeometry = DEFAULT_CUBBY_GEOMETRY) -> None:
    """Write the cubby as a kinematic rigid body: two side walls, a back wall and a shelf board on top.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Interior dimensions and wall friction of the cubby.
    """
    from pxr import UsdPhysics

    g = geometry
    stage, root = _new_stage(path, "CubbyShelf")
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    material = _define_physics_material(
        stage, "/CubbyShelf/PhysicsMaterials/cubby_wood", g.static_friction, g.dynamic_friction
    )
    half_width = g.interior_width_m / 2
    outer_depth = g.interior_depth_m + g.wall_thickness_m
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        y_inner, y_outer = side * half_width, side * (half_width + g.wall_thickness_m)
        _add_box(
            stage,
            f"/CubbyShelf/{side_name}_wall",
            (0.0, min(y_inner, y_outer), 0.0),
            (outer_depth, max(y_inner, y_outer), g.interior_height_m),
            _CUBBY_WALL,
            material,
        )
    _add_box(
        stage,
        "/CubbyShelf/back_wall",
        (g.interior_depth_m, -half_width, 0.0),
        (outer_depth, half_width, g.interior_height_m),
        _CUBBY_WALL,
        material,
    )
    top_half_width = half_width + g.wall_thickness_m + g.top_overhang_m
    _add_box(
        stage,
        "/CubbyShelf/shelf_board",
        (g.top_setback_m, -top_half_width, g.interior_height_m),
        (outer_depth + 0.01, top_half_width, g.interior_height_m + g.top_thickness_m),
        _CUBBY_TOP,
        material,
    )
    stage.GetRootLayer().Save()


def get_cubby_shelf_usd_path(geometry: CubbyShelfGeometry = DEFAULT_CUBBY_GEOMETRY) -> str:
    """Return a cached USD path for the cubby, writing the file on first use."""
    return _cached_usd_path("cubby_shelf", repr(geometry), lambda path: write_cubby_shelf_usd(path, geometry))


@register_asset
class CubbyShelf(Object):
    """Static cubby under a shelf board, 10 mm wider than the sealed crate on each side; see the module frames."""

    registry_name = "cubby_shelf"
    tags = ["object", "fixture", "procedural", "crate_cubby_push"]
    geometry = DEFAULT_CUBBY_GEOMETRY

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the cubby.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the cubby frame (entrance center on the floor); its rotation must be a yaw only.
        """
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            usd_path=get_cubby_shelf_usd_path(self.geometry),
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)},
        )


@register_asset
class RubberFloorMat(Object):
    """Static rubber floor mat (0.7 static, 0.6 dynamic friction) that the crate slides on, inside and outside the cubby."""

    registry_name = "rubber_floor_mat"
    tags = ["object", "fixture", "procedural", "crate_cubby_push"]

    def __init__(
        self,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        size_xyz_m: tuple[float, float, float] = (0.38, 0.2, 0.006),
        static_friction: float = 0.7,
        dynamic_friction: float = 0.6,
    ):
        """Create the floor mat.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Pose of the mat's center.
            size_xyz_m: Mat size; the last entry is its thickness.
            static_friction: Static friction coefficient of the mat surface.
            dynamic_friction: Dynamic friction coefficient of the mat surface.
        """
        self.thickness_m = float(size_xyz_m[2])
        spawner_cfg = sim_utils.CuboidCfg(
            size=tuple(float(value) for value in size_xyz_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.20, 0.18), roughness=0.95),
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
class PushProbe(Object):
    """Push rod with a 2 cm square tip that stands in for an arm-held fingertip in object-control pushes.

    The rod is 8 cm long along X and weighs 1 kg so the runner's orientation controller can hold it as stiffly as an
    arm holds a fingertip; its 2 x 2 cm end face is the contact patch.
    """

    registry_name = "push_probe"
    tags = ["object", "procedural", "object_control", "crate_cubby_push"]

    def __init__(self, name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None):
        """Create the probe.

        Args:
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
        """
        spawner_cfg = sim_utils.CuboidCfg(
            size=(0.08, 0.02, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, sleep_threshold=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.10), roughness=0.5),
        )
        super().__init__(
            name=name if name is not None else self.registry_name,
            prim_path=prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=initial_pose,
            tags=list(self.tags),
        )
