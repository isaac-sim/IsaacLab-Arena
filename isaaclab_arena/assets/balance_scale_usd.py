# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD for a passive equal-arm balance scale.

The scale is a fixed-base articulation of four links: a stand, a beam on a revolute pivot, and two pans hanging
from the beam ends on revolute pivots. Only gravity, contacts and the joint damping set by the asset's actuators act
on it. The beam's mass sits below the fulcrum, and the pan pivots sit slightly below it, so the beam rests level when
both pans carry equal mass and tilts toward the heavier pan otherwise.

Asset frame: Z up, origin at the bottom center of the stand. The beam lies along Y and turns about X. The left pan
hangs at +Y and the right pan at -Y, seen by a robot standing on the -X side. The pans sit in front of the beam line
(toward -X) on rear hangers, so the space above each pan stays open.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

BEAM_JOINT_NAME = "beam_pivot"
LEFT_PAN_JOINT_NAME = "left_pan_pivot"
RIGHT_PAN_JOINT_NAME = "right_pan_pivot"
LEFT_PAN_BODY_NAME = "left_pan"
RIGHT_PAN_BODY_NAME = "right_pan"
BEAM_BODY_NAME = "beam"
STAND_BODY_NAME = "stand"


@dataclass(frozen=True)
class BalanceScaleGeometry:
    """Dimensions (m), masses (kg) and joint limits (deg) of the procedural balance scale."""

    fulcrum_height_m: float = 0.24
    """Height of the beam pivot above the stand's bottom face."""

    half_arm_m: float = 0.15
    """Distance from the fulcrum to each pan pivot along the beam."""

    pan_pivot_drop_m: float = 0.005
    """Distance of the pan pivots below the fulcrum; adds load-dependent restoring torque."""

    beam_mass_kg: float = 0.18
    beam_center_of_mass_drop_m: float = 0.03
    """Distance of the beam's center of mass below the fulcrum; sets the unloaded restoring torque."""

    pan_mass_kg: float = 0.09
    pan_hanger_length_m: float = 0.14
    """Vertical distance from a pan pivot down to the top of the pan floor."""

    pan_forward_offset_m: float = 0.075
    """Distance of the pan center in front of (toward -X) the beam line."""

    pan_floor_size_m: float = 0.13
    pan_floor_thickness_m: float = 0.004
    pan_rim_height_m: float = 0.022
    pan_rim_thickness_m: float = 0.004

    stand_mass_kg: float = 2.0
    beam_limit_deg: float = 10.0
    pan_limit_deg: float = 25.0


DEFAULT_GEOMETRY = BalanceScaleGeometry()

_BRASS = (0.72, 0.55, 0.22)
_STEEL = (0.62, 0.64, 0.68)
_GRAPHITE = (0.16, 0.17, 0.19)
_RED = (0.75, 0.12, 0.08)


def _set_local_translation(prim: Usd.Prim, t_P_C: tuple[float, float, float]) -> None:
    """Author a translate op that places prim C in its parent frame P."""
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*t_P_C))


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


def _add_cylinder(
    stage: Usd.Stage,
    path: str,
    center_L: tuple[float, float, float],
    radius: float,
    height: float,
    axis: str,
    color: tuple[float, float, float],
    collide: bool = False,
) -> None:
    """Add a cylinder to a link at a center expressed in the link frame L."""
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr(axis)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*center_L))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())


def _define_link(stage: Usd.Stage, path: str, t_A_L: tuple[float, float, float], mass_kg: float) -> Usd.Prim:
    """Define a rigid link whose frame L sits at t_A_L in the asset frame A."""
    link = UsdGeom.Xform.Define(stage, path).GetPrim()
    _set_local_translation(link, t_A_L)
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(mass_kg)
    return link


def _define_revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    t_B0_J: tuple[float, float, float],
    limit_deg: float,
) -> None:
    """Define a damped-only revolute joint about X, anchored at t_B0_J in body0 and at body1's origin."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*t_B0_J))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateLowerLimitAttr(-limit_deg)
    joint.CreateUpperLimitAttr(limit_deg)
    # A drive with zero gains; the asset's actuator configuration supplies the bearing damping.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)


def _add_pan_geometry(stage: Usd.Stage, pan_path: str, geometry: BalanceScaleGeometry) -> None:
    """Add a rear hanger and an open dish below a pan pivot, in the pan frame."""
    g = geometry
    floor_top_z = -g.pan_hanger_length_m
    floor_center_z = floor_top_z - g.pan_floor_thickness_m / 2
    center_x = -g.pan_forward_offset_m
    half = g.pan_floor_size_m / 2
    rim_center_z = floor_top_z + g.pan_rim_height_m / 2
    rim_t = g.pan_rim_thickness_m
    _add_box(
        stage,
        f"{pan_path}/floor",
        (center_x, 0.0, floor_center_z),
        (2 * half, 2 * half, g.pan_floor_thickness_m),
        _BRASS,
    )
    # Four separate rim walls keep the dish open; one convex hull over it would make loads float on its lid.
    _add_box(
        stage,
        f"{pan_path}/rim_front",
        (center_x - half + rim_t / 2, 0.0, rim_center_z),
        (rim_t, 2 * half, g.pan_rim_height_m),
        _BRASS,
    )
    _add_box(
        stage,
        f"{pan_path}/rim_back",
        (center_x + half - rim_t / 2, 0.0, rim_center_z),
        (rim_t, 2 * half, g.pan_rim_height_m),
        _BRASS,
    )
    _add_box(
        stage,
        f"{pan_path}/rim_left",
        (center_x, half - rim_t / 2, rim_center_z),
        (2 * half, rim_t, g.pan_rim_height_m),
        _BRASS,
    )
    _add_box(
        stage,
        f"{pan_path}/rim_right",
        (center_x, -half + rim_t / 2, rim_center_z),
        (2 * half, rim_t, g.pan_rim_height_m),
        _BRASS,
    )
    # Rear hanger: a post from the pivot down behind the dish, and a strut joining it to the back rim.
    back_x = center_x + half
    post_x = back_x + 0.006
    _add_box(stage, f"{pan_path}/hanger_post", (post_x, 0.0, floor_top_z / 2), (0.008, 0.012, abs(floor_top_z)), _STEEL)
    _add_box(
        stage,
        f"{pan_path}/hanger_strut",
        ((back_x + post_x) / 2, 0.0, floor_top_z + 0.004),
        (0.014, 0.012, 0.008),
        _STEEL,
    )
    _add_cylinder(stage, f"{pan_path}/hook", (post_x, 0.0, 0.0), 0.006, 0.02, "X", _STEEL)


def write_balance_scale_usd(path: Path, geometry: BalanceScaleGeometry = DEFAULT_GEOMETRY) -> None:
    """Write the balance scale articulation to a USD file.

    Args:
        path: Output USD path; overwritten if it exists.
        geometry: Dimensions, masses and joint limits of the scale.
    """
    g = geometry
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/BalanceScale").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # Stand: base plate and a column behind the beam plane. The beam links to it by a joint, so they never collide.
    stand_path = f"/BalanceScale/{STAND_BODY_NAME}"
    _define_link(stage, stand_path, (0.0, 0.0, 0.0), g.stand_mass_kg)
    _add_box(stage, f"{stand_path}/base", (0.01, 0.0, 0.01), (0.12, 0.09, 0.02), _GRAPHITE)
    column_height = g.fulcrum_height_m + 0.01 - 0.02
    _add_box(
        stage, f"{stand_path}/column", (0.03, 0.0, 0.02 + column_height / 2), (0.018, 0.018, column_height), _GRAPHITE
    )
    _add_cylinder(stage, f"{stand_path}/axle", (0.01, 0.0, g.fulcrum_height_m), 0.004, 0.05, "X", _STEEL)
    # Index plate behind the pointer tip, marking level.
    _add_box(
        stage,
        f"{stand_path}/index_plate",
        (0.02, 0.0, g.fulcrum_height_m - 0.105),
        (0.004, 0.05, 0.016),
        _STEEL,
        collide=False,
    )
    _add_box(
        stage,
        f"{stand_path}/index_mark",
        (0.0175, 0.0, g.fulcrum_height_m - 0.105),
        (0.002, 0.002, 0.018),
        _RED,
        collide=False,
    )

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/BalanceScale/root_joint")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(stand_path)])

    # Beam: crossbar, pointer and trim bob. Its center of mass is authored explicitly below the fulcrum.
    beam_path = f"/BalanceScale/{BEAM_BODY_NAME}"
    beam = _define_link(stage, beam_path, (0.0, 0.0, g.fulcrum_height_m), g.beam_mass_kg)
    beam_length = 2 * g.half_arm_m + 0.03
    _add_box(stage, f"{beam_path}/crossbar", (0.0, 0.0, 0.0), (0.012, beam_length, 0.016), _BRASS)
    _add_box(stage, f"{beam_path}/pointer", (-0.006, 0.0, -0.05), (0.003, 0.003, 0.1), _GRAPHITE, collide=False)
    _add_cylinder(stage, f"{beam_path}/trim_bob", (-0.006, 0.0, -0.06), 0.009, 0.01, "X", _BRASS, collide=False)
    beam_mass_api = UsdPhysics.MassAPI(beam)
    beam_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -g.beam_center_of_mass_drop_m))
    beam_inertia_x = g.beam_mass_kg * (beam_length**2 + 0.016**2) / 12.0
    beam_mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(beam_inertia_x, 2e-4, beam_inertia_x))
    beam_mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

    pan_pivot_z = g.fulcrum_height_m - g.pan_pivot_drop_m
    for pan_name, side in ((LEFT_PAN_BODY_NAME, 1.0), (RIGHT_PAN_BODY_NAME, -1.0)):
        pan_path = f"/BalanceScale/{pan_name}"
        _define_link(stage, pan_path, (0.0, side * g.half_arm_m, pan_pivot_z), g.pan_mass_kg)
        _add_pan_geometry(stage, pan_path, g)

    _define_revolute_joint(
        stage,
        f"/BalanceScale/{BEAM_JOINT_NAME}",
        stand_path,
        beam_path,
        (0.0, 0.0, g.fulcrum_height_m),
        g.beam_limit_deg,
    )
    for joint_name, pan_name, side in (
        (LEFT_PAN_JOINT_NAME, LEFT_PAN_BODY_NAME, 1.0),
        (RIGHT_PAN_JOINT_NAME, RIGHT_PAN_BODY_NAME, -1.0),
    ):
        _define_revolute_joint(
            stage,
            f"/BalanceScale/{joint_name}",
            beam_path,
            f"/BalanceScale/{pan_name}",
            (0.0, side * g.half_arm_m, -g.pan_pivot_drop_m),
            g.pan_limit_deg,
        )
    stage.GetRootLayer().Save()


def get_balance_scale_usd_path(geometry: BalanceScaleGeometry = DEFAULT_GEOMETRY) -> str:
    """Return a cached USD path for the balance scale, writing the file on first use."""
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr(geometry).encode()).hexdigest()[:16]
    path = cache_dir / f"balance_scale_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_balance_scale_usd(tmp_path, geometry)
        os.replace(tmp_path, path)
    return str(path)
