# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD for a light three-compartment cabinet whose fronts open by hidden modes, and for a tray.

The cabinet is a floating-base articulation (optionally fixed): a carcass link with three stacked compartments and
one front panel per compartment. Every front looks the same; only its joint differs with its mode:

* ``slide_left``: prismatic along Y, opening toward +Y (the robot's left, seen from the -X side), limits
  [0, travel] m.
* ``slide_right``: prismatic along Y, opening toward -Y, limits [-travel, 0] m.
* ``fold_down``: revolute about Y at the front's outer bottom edge, limits [-95, 0] deg; negative angles tip the
  top of the front toward -X and down. Its center of mass sits behind the hinge, so gravity holds it shut.

Every front is closed at joint position 0. Asset frame: Z up, origin at the bottom center of the carcass, fronts on
the -X face. The compartments stand on a plinth; compartment 1 is the bottom one.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

FRONT_MODES = ("slide_left", "slide_right", "fold_down")
NUM_COMPARTMENTS = 3
CARCASS_BODY_NAME = "carcass"


def front_joint_name(index: int) -> str:
    """Return the joint name of compartment ``index``'s front (1 is the bottom compartment)."""
    return f"front_{index}"


def front_body_name(index: int) -> str:
    """Return the body name of compartment ``index``'s front (1 is the bottom compartment)."""
    return f"front_panel_{index}"


@dataclass(frozen=True)
class CabinetGeometry:
    """Dimensions (m), masses (kg) and travel limits of the procedural cabinet."""

    depth_m: float = 0.12
    width_m: float = 0.18
    wall_m: float = 0.008
    plinth_height_m: float = 0.05
    """Height of the plinth under the bottom compartment, which keeps a hand reaching into it clear of the table."""

    compartment_height_m: float = 0.08
    front_thickness_m: float = 0.01
    front_gap_m: float = 0.002
    """Gap between neighbouring fronts and between a front and the carcass edges."""

    slide_travel_m: float = 0.18
    fold_limit_deg: float = 95.0
    carcass_mass_kg: float = 1.2
    front_mass_kg: float = 0.12

    @property
    def height_m(self) -> float:
        return (
            self.plinth_height_m + (NUM_COMPARTMENTS + 1) * self.wall_m + NUM_COMPARTMENTS * self.compartment_height_m
        )

    def compartment_floor_z(self, index: int) -> float:
        """Return the height of compartment ``index``'s floor (1 is the bottom compartment)."""
        return self.plinth_height_m + self.wall_m + (index - 1) * (self.compartment_height_m + self.wall_m)

    def front_bottom_z(self, index: int) -> float:
        """Return the height of the bottom edge of compartment ``index``'s front, which carries the fold hinge."""
        return self.compartment_floor_z(index) - self.wall_m / 2 + self.front_gap_m

    @property
    def front_height_m(self) -> float:
        return self.compartment_height_m + self.wall_m - 2 * self.front_gap_m

    @property
    def front_outer_x(self) -> float:
        """X of the fronts' outer face, which carries the fold hinges."""
        return -self.depth_m / 2 - 0.001 - self.front_thickness_m


DEFAULT_CABINET_GEOMETRY = CabinetGeometry()


@dataclass(frozen=True)
class TrayGeometry:
    """Dimensions (m) and mass (kg) of the procedural tray."""

    length_x_m: float = 0.14
    width_y_m: float = 0.16
    floor_m: float = 0.006
    rim_height_m: float = 0.02
    rim_m: float = 0.006
    mass_kg: float = 0.25


DEFAULT_TRAY_GEOMETRY = TrayGeometry()

_WOOD = (0.62, 0.45, 0.30)
_FRONT = (0.80, 0.78, 0.72)
_HANDLE = (0.25, 0.25, 0.27)
_TRAY = (0.20, 0.35, 0.55)


def _add_box(
    stage: Usd.Stage,
    path: str,
    center_L: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    collide: bool = True,
) -> None:
    """Add a box to a link at a center expressed in the link frame L."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center_L))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _define_link(stage: Usd.Stage, path: str, t_A_L: tuple[float, float, float], mass_kg: float) -> Usd.Prim:
    """Define a rigid link whose frame L sits at t_A_L in the asset frame A."""
    link = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(*t_A_L))
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(mass_kg)
    return link


def _new_stage(path: Path, root_path: str) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def _add_carcass(stage: Usd.Stage, carcass_path: str, g: CabinetGeometry) -> None:
    """Add the carcass panels (open toward -X) to the carcass link, whose frame equals the asset frame."""
    d, w, t, h, p = g.depth_m, g.width_m, g.wall_m, g.height_m, g.plinth_height_m
    inner_h = h - p - 2 * t
    inner_z = (p + h) / 2
    _add_box(stage, f"{carcass_path}/plinth", (0.005, 0.0, p / 2), (d - 0.01, w - 0.02, p), _WOOD)
    _add_box(stage, f"{carcass_path}/bottom", (0.0, 0.0, p + t / 2), (d, w, t), _WOOD)
    _add_box(stage, f"{carcass_path}/top", (0.0, 0.0, h - t / 2), (d, w, t), _WOOD)
    _add_box(stage, f"{carcass_path}/side_left", (0.0, w / 2 - t / 2, inner_z), (d, t, inner_h), _WOOD)
    _add_box(stage, f"{carcass_path}/side_right", (0.0, -w / 2 + t / 2, inner_z), (d, t, inner_h), _WOOD)
    _add_box(stage, f"{carcass_path}/back", (d / 2 - t / 2, 0.0, inner_z), (t, w - 2 * t, inner_h), _WOOD)
    for index in range(2, NUM_COMPARTMENTS + 1):
        shelf_z = g.compartment_floor_z(index) - t / 2
        _add_box(stage, f"{carcass_path}/shelf_{index}", (0.0, 0.0, shelf_z), (d - t, w - 2 * t, t), _WOOD)


def _add_front(stage: Usd.Stage, front_path: str, g: CabinetGeometry) -> None:
    """Add a front panel and its handle in the front's link frame, whose origin is the outer bottom edge center."""
    tf, hf = g.front_thickness_m, g.front_height_m
    width = g.width_m - 2 * g.front_gap_m
    _add_box(stage, f"{front_path}/panel", (tf / 2, 0.0, hf / 2), (tf, width, hf), _FRONT)
    _add_box(stage, f"{front_path}/handle", (-0.015, 0.0, hf / 2), (0.03, 0.05, 0.012), _HANDLE)
    # Center of mass behind the hinge line, so gravity holds a fold-down front against its closed stop.
    front = stage.GetPrimAtPath(front_path)
    mass_api = UsdPhysics.MassAPI(front)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(tf / 2, 0.0, hf / 2))
    m = g.front_mass_kg
    mass_api.CreateDiagonalInertiaAttr(
        Gf.Vec3f(m * (width**2 + hf**2) / 12.0, m * (tf**2 + hf**2) / 12.0, m * (tf**2 + width**2) / 12.0)
    )
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))


def _define_front_joint(
    stage: Usd.Stage, path: str, carcass_path: str, front_path: str, t_C_J: tuple[float, float, float], mode: str, g
) -> None:
    """Define the joint of one front for its opening mode, anchored at t_C_J in the carcass frame C."""
    if mode == "fold_down":
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        joint.CreateLowerLimitAttr(-g.fold_limit_deg)
        joint.CreateUpperLimitAttr(0.0)
        drive_type = "angular"
    else:
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        if mode == "slide_left":
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(g.slide_travel_m)
        else:
            assert mode == "slide_right", f"Unknown front mode '{mode}'"
            joint.CreateLowerLimitAttr(-g.slide_travel_m)
            joint.CreateUpperLimitAttr(0.0)
        drive_type = "linear"
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carcass_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(front_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*t_C_J))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    # A drive with zero gains; the asset's actuator configuration supplies the joint damping.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_type)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)


def write_sliding_front_cabinet_usd(
    path: Path,
    front_modes: tuple[str, ...],
    fixed_base: bool = False,
    geometry: CabinetGeometry = DEFAULT_CABINET_GEOMETRY,
) -> None:
    """Write the cabinet articulation to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        front_modes: Opening mode of fronts 1, 2 and 3 (bottom to top), each one of FRONT_MODES.
        fixed_base: Whether to fix the carcass to the world.
        geometry: Dimensions, masses and limits of the cabinet.
    """
    assert len(front_modes) == NUM_COMPARTMENTS, f"Give one mode per front, got {front_modes}"
    g = geometry
    stage, root = _new_stage(path, "/SlidingFrontCabinet")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    carcass_path = f"/SlidingFrontCabinet/{CARCASS_BODY_NAME}"
    _define_link(stage, carcass_path, (0.0, 0.0, 0.0), g.carcass_mass_kg)
    _add_carcass(stage, carcass_path, g)
    if fixed_base:
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/SlidingFrontCabinet/root_joint")
        fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(carcass_path)])
    for index, mode in enumerate(front_modes, start=1):
        front_path = f"/SlidingFrontCabinet/{front_body_name(index)}"
        t_C_hinge = (g.front_outer_x, 0.0, g.front_bottom_z(index))
        _define_link(stage, front_path, t_C_hinge, g.front_mass_kg)
        _add_front(stage, front_path, g)
        _define_front_joint(
            stage, f"/SlidingFrontCabinet/{front_joint_name(index)}", carcass_path, front_path, t_C_hinge, mode, g
        )
    stage.GetRootLayer().Save()


def write_tray_usd(path: Path, geometry: TrayGeometry = DEFAULT_TRAY_GEOMETRY) -> None:
    """Write a rigid open tray (a floor and four rim walls) with its origin at the bottom center."""
    g = geometry
    stage, root = _new_stage(path, "/Tray")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(g.mass_kg)
    lx, wy, rim, rh = g.length_x_m, g.width_y_m, g.rim_m, g.rim_height_m
    _add_box(stage, "/Tray/floor", (0.0, 0.0, g.floor_m / 2), (lx, wy, g.floor_m), _TRAY)
    rim_z = g.floor_m + rh / 2
    # Separate rim walls keep the tray open; one convex hull over it would make tokens rest on its lid.
    _add_box(stage, "/Tray/rim_front", (-lx / 2 + rim / 2, 0.0, rim_z), (rim, wy, rh), _TRAY)
    _add_box(stage, "/Tray/rim_back", (lx / 2 - rim / 2, 0.0, rim_z), (rim, wy, rh), _TRAY)
    _add_box(stage, "/Tray/rim_left", (0.0, wy / 2 - rim / 2, rim_z), (lx - 2 * rim, rim, rh), _TRAY)
    _add_box(stage, "/Tray/rim_right", (0.0, -wy / 2 + rim / 2, rim_z), (lx - 2 * rim, rim, rh), _TRAY)
    stage.GetRootLayer().Save()


def _cached_usd_path(stem: str, key: str, write) -> str:
    """Return a USD path in the procedural cache, keyed by content, writing it atomically on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache_dir / f"{stem}_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write(tmp_path)
        os.replace(tmp_path, path)
    return str(path)


def get_sliding_front_cabinet_usd_path(
    front_modes: tuple[str, ...],
    fixed_base: bool = False,
    geometry: CabinetGeometry = DEFAULT_CABINET_GEOMETRY,
) -> str:
    """Return a cached USD path for the cabinet with the given front modes, writing the file on first use."""
    key = repr((tuple(front_modes), fixed_base, geometry))
    return _cached_usd_path(
        "sliding_front_cabinet",
        key,
        lambda path: write_sliding_front_cabinet_usd(path, tuple(front_modes), fixed_base, geometry),
    )


def get_tray_usd_path(geometry: TrayGeometry = DEFAULT_TRAY_GEOMETRY) -> str:
    """Return a cached USD path for the tray, writing the file on first use."""
    return _cached_usd_path("procedural_tray", repr(geometry), lambda path: write_tray_usd(path, geometry))
