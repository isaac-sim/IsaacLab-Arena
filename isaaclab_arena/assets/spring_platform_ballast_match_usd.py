# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD for a rack of three look-alike spring platforms.

The rack is a fixed-base articulation: a base link carrying three guide housings and three dial boards, and three
platform tops, each on a prismatic Z joint. The USD authors zero-gain drives; the asset's actuators supply each
spring's stiffness and damping, so all three platforms share one USD whatever their spring rates.

Asset frame: Z up, origin at the bottom center of the base plate. The platforms stand in a row along Y, platform 0
at +Y (the robot's left, seen from the -X side), platform 2 at -Y. Each top's link frame sits at the center of its
top surface, so a joint position of zero puts every top at the same level height, and the negated joint position is
the deflection. A red needle on each top moves along a dial board in front of it (toward -X), which carries a black
level mark and a green target band.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

NUM_PLATFORMS = 3
BASE_BODY_NAME = "base"


def platform_top_body_name(index: int) -> str:
    """Return the body name of platform ``index``'s top."""
    return f"top_{index}"


def platform_joint_name(index: int) -> str:
    """Return the prismatic joint name of platform ``index``."""
    return f"spring_{index}"


@dataclass(frozen=True)
class SpringPlatformGeometry:
    """Dimensions (m), masses (kg) and travel limits (m) of the procedural spring-platform rack."""

    platform_pitch_m: float = 0.15
    """Distance along Y between neighbouring platform centers."""

    level_height_m: float = 0.11
    """Height of every top surface above the base plate's bottom face at zero deflection."""

    top_size_m: float = 0.09
    top_thickness_m: float = 0.012
    top_mass_kg: float = 0.08
    """Mass of each moving top; the spring's rest offset cancels its weight."""

    lower_limit_m: float = -0.05
    """Lowest joint position (end stop), relative to level."""

    upper_limit_m: float = 0.01

    base_mass_kg: float = 3.0
    housing_size_m: float = 0.05
    housing_height_m: float = 0.033

    target_deflection_m: float = 0.022
    """Center of the green band painted on each dial board."""

    band_half_width_m: float = 0.005
    """Half height of the green band; the task's default tolerance matches it."""


DEFAULT_GEOMETRY = SpringPlatformGeometry()

_STEEL = (0.62, 0.64, 0.68)
_GRAPHITE = (0.16, 0.17, 0.19)
_TOP = (0.30, 0.42, 0.58)
_DIAL = (0.92, 0.92, 0.88)
_GREEN = (0.10, 0.70, 0.20)
_RED = (0.80, 0.10, 0.08)
_BLACK = (0.02, 0.02, 0.02)


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


def _add_platform_housing(stage: Usd.Stage, base_path: str, index: int, y_A: float, g: SpringPlatformGeometry) -> None:
    """Add a guide housing and a dial board for one platform to the base link (frame equals asset frame)."""
    plate_top_z = 0.012
    wall_t = 0.006
    half = g.housing_size_m / 2
    wall_z = plate_top_z + g.housing_height_m / 2
    prefix = f"{base_path}/platform_{index}"
    for name, center_xy, size_xy in (
        ("housing_front", (-half + wall_t / 2, 0.0), (wall_t, g.housing_size_m)),
        ("housing_back", (half - wall_t / 2, 0.0), (wall_t, g.housing_size_m)),
        ("housing_left", (0.0, half - wall_t / 2), (g.housing_size_m, wall_t)),
        ("housing_right", (0.0, -half + wall_t / 2), (g.housing_size_m, wall_t)),
    ):
        _add_box(
            stage,
            f"{prefix}_{name}",
            (center_xy[0], y_A + center_xy[1], wall_z),
            (size_xy[0], size_xy[1], g.housing_height_m),
            _GRAPHITE,
        )
    # Dial board in front of the top; the needle passes through its (implied) slot. Visual only.
    board_front_x = -g.top_size_m / 2 - 0.017
    board_t = 0.004
    board_bottom_z = plate_top_z
    board_top_z = g.level_height_m + 0.012
    board_width = 0.03
    _add_box(
        stage,
        f"{prefix}_dial",
        (board_front_x + board_t / 2, y_A, (board_bottom_z + board_top_z) / 2),
        (board_t, board_width, board_top_z - board_bottom_z),
        _DIAL,
        collide=False,
    )
    needle_level_z = g.level_height_m - g.top_thickness_m / 2
    _add_box(
        stage,
        f"{prefix}_band",
        (board_front_x - 0.0005, y_A, needle_level_z - g.target_deflection_m),
        (0.001, board_width, 2 * g.band_half_width_m),
        _GREEN,
        collide=False,
    )
    _add_box(
        stage,
        f"{prefix}_level_mark",
        (board_front_x - 0.0005, y_A, needle_level_z),
        (0.001, board_width, 0.0015),
        _BLACK,
        collide=False,
    )


def _add_platform_top(stage: Usd.Stage, top_path: str, g: SpringPlatformGeometry) -> None:
    """Add the top plate, its guide stem and its needle, in the top's link frame (origin at the top surface)."""
    _add_box(
        stage,
        f"{top_path}/plate",
        (0.0, 0.0, -g.top_thickness_m / 2),
        (g.top_size_m, g.top_size_m, g.top_thickness_m),
        _TOP,
    )
    stem_length = 0.07
    _add_box(
        stage,
        f"{top_path}/stem",
        (0.0, 0.0, -g.top_thickness_m - stem_length / 2),
        (0.018, 0.018, stem_length),
        _STEEL,
        collide=False,
    )
    needle_start_x = -g.top_size_m / 2
    needle_tip_x = -g.top_size_m / 2 - 0.024
    _add_box(
        stage,
        f"{top_path}/needle",
        ((needle_start_x + needle_tip_x) / 2, 0.0, -g.top_thickness_m / 2),
        (needle_start_x - needle_tip_x, 0.004, 0.003),
        _RED,
        collide=False,
    )
    # Explicit inertia of the plate; the visual-only stem and needle add no mass.
    top = stage.GetPrimAtPath(top_path)
    mass_api = UsdPhysics.MassAPI(top)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -g.top_thickness_m / 2))
    inertia_xy = g.top_mass_kg * (g.top_size_m**2 + g.top_thickness_m**2) / 12.0
    inertia_z = g.top_mass_kg * (2 * g.top_size_m**2) / 12.0
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_xy, inertia_xy, inertia_z))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))


def platform_y_A(index: int, geometry: SpringPlatformGeometry = DEFAULT_GEOMETRY) -> float:
    """Return platform ``index``'s center along Y in the asset frame."""
    return (1 - index) * geometry.platform_pitch_m


def write_spring_platform_rack_usd(path: Path, geometry: SpringPlatformGeometry = DEFAULT_GEOMETRY) -> None:
    """Write the spring-platform rack articulation to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions, masses and travel limits of the rack.
    """
    g = geometry
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/SpringPlatformRack").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_path = f"/SpringPlatformRack/{BASE_BODY_NAME}"
    _define_link(stage, base_path, (0.0, 0.0, 0.0), g.base_mass_kg)
    rack_width = 2 * g.platform_pitch_m + g.top_size_m + 0.03
    _add_box(stage, f"{base_path}/plate", (-0.0075, 0.0, 0.006), (0.125, rack_width, 0.012), _GRAPHITE)
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/SpringPlatformRack/root_joint")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])

    for index in range(NUM_PLATFORMS):
        y_A = platform_y_A(index, g)
        _add_platform_housing(stage, base_path, index, y_A, g)
        top_path = f"/SpringPlatformRack/{platform_top_body_name(index)}"
        _define_link(stage, top_path, (0.0, y_A, g.level_height_m), g.top_mass_kg)
        _add_platform_top(stage, top_path, g)

        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/SpringPlatformRack/{platform_joint_name(index)}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(top_path)])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, y_A, g.level_height_m))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateLowerLimitAttr(g.lower_limit_m)
        joint.CreateUpperLimitAttr(g.upper_limit_m)
        # A drive with zero gains; the asset's actuator configuration supplies the spring and its damping.
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    stage.GetRootLayer().Save()


def get_spring_platform_rack_usd_path(geometry: SpringPlatformGeometry = DEFAULT_GEOMETRY) -> str:
    """Return a cached USD path for the spring-platform rack, writing the file on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr(geometry).encode()).hexdigest()[:16]
    path = cache_dir / f"spring_platform_rack_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_spring_platform_rack_usd(tmp_path, geometry)
        os.replace(tmp_path, path)
    return str(path)
