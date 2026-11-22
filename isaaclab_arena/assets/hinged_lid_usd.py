# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD building blocks for containers with a lid on a revolute joint.

Shared by assets whose lid turns about a hidden edge or corner, such as a hinged house roof or a swivel cover. Every
helper writes plain ``UsdPhysics`` schemas only, so files can be written without Kit's PhysX schema module. Geometry is
built from boxes and cylinders with per-prim display colors.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

Color = tuple[float, float, float]
Vector3 = tuple[float, float, float]


def create_stage(path: Path, root_name: str) -> tuple[Usd.Stage, Usd.Prim]:
    """Create a Z-up, meter, kilogram USD stage with an Xform default prim.

    Args:
        path: Output USD path; overwritten if it exists.
        root_name: Name of the root Xform prim.

    Returns:
        The stage and its root prim.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def add_box(
    stage: Usd.Stage,
    path: str,
    center_L: Vector3,
    size_xyz: Vector3,
    color: Color,
    collide: bool = True,
) -> Usd.Prim:
    """Add an axis-aligned box centered at center_L in its parent link frame L."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center_L))
    xform.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def add_cylinder(
    stage: Usd.Stage,
    path: str,
    center_L: Vector3,
    radius: float,
    height: float,
    axis: str,
    color: Color,
    collide: bool = False,
) -> Usd.Prim:
    """Add a cylinder centered at center_L in its parent link frame L."""
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr(axis)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*center_L))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder.GetPrim()


def add_sphere(
    stage: Usd.Stage,
    path: str,
    center_L: Vector3,
    radius: float,
    color: Color,
    collide: bool = False,
) -> Usd.Prim:
    """Add a sphere centered at center_L in its parent link frame L."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*center_L))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere.GetPrim()


def define_link(stage: Usd.Stage, path: str, t_A_L: Vector3, mass_kg: float) -> Usd.Prim:
    """Define a rigid link whose frame L sits at t_A_L in its parent frame A.

    PhysX derives the center of mass and inertia from the link's collision shapes, scaled to mass_kg.
    """
    link = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(*t_A_L))
    UsdPhysics.RigidBodyAPI.Apply(link)
    UsdPhysics.MassAPI.Apply(link).CreateMassAttr(mass_kg)
    return link


def fix_link_to_world(stage: Usd.Stage, path: str, link_path: str) -> None:
    """Fix a link to the world with a FixedJoint, making its articulation fixed-base."""
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, path)
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])


def define_revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    t_B0_J: Vector3,
    t_B1_J: Vector3,
    q_B_J: Gf.Quatf,
    lower_limit_deg: float,
    upper_limit_deg: float,
) -> None:
    """Define a revolute joint about the X axis of joint frame J, with a zero-gain angular drive.

    Both bodies share the joint frame orientation q_B_J at joint position zero, so their relative pose there is the one
    authored on the links. The asset's actuator configuration supplies stiffness and damping.

    Args:
        stage: Stage to author on.
        path: Joint prim path.
        body0: Parent link path.
        body1: Child link path.
        t_B0_J: Joint origin in the parent link frame.
        t_B1_J: Joint origin in the child link frame.
        q_B_J: Rotation from joint frame J into both link frames; J's X axis is the rotation axis.
        lower_limit_deg: Lower joint limit, in degrees.
        upper_limit_deg: Upper joint limit, in degrees.
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*t_B0_J))
    joint.CreateLocalRot0Attr(q_B_J)
    joint.CreateLocalPos1Attr(Gf.Vec3f(*t_B1_J))
    joint.CreateLocalRot1Attr(q_B_J)
    joint.CreateLowerLimitAttr(lower_limit_deg)
    joint.CreateUpperLimitAttr(upper_limit_deg)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)


def filter_collisions(link_paths: list[str], stage: Usd.Stage) -> None:
    """Disable collisions between every pair of the given links."""
    for index, link_path in enumerate(link_paths):
        others = [Sdf.Path(other) for other in link_paths[index + 1 :]]
        if others:
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link_path)).CreateFilteredPairsRel().SetTargets(
                others
            )


def get_cached_usd_path(prefix: str, content_key: object, write_usd: Callable[[Path], None]) -> str:
    """Return a cached USD path for procedurally generated content, writing the file on first use.

    Args:
        prefix: File name prefix in the cache directory.
        content_key: Value whose repr identifies the file content; a different repr writes a different file.
        write_usd: Function writing the USD to a given path.

    Returns:
        Path of the cached USD file.
    """
    cache_dir = Path(os.environ.get("ISAACLAB_ARENA_PROCEDURAL_CACHE", Path(tempfile.gettempdir()) / "isaaclab_arena"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr(content_key).encode()).hexdigest()[:16]
    path = cache_dir / f"{prefix}_{digest}.usda"
    if not path.exists():
        # Write to a temporary file first so parallel processes never read a partial file.
        tmp_path = path.with_suffix(f".{os.getpid()}.usda")
        write_usd(tmp_path)
        os.replace(tmp_path, path)
    return str(path)
