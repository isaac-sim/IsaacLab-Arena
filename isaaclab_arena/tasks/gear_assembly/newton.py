# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton compatibility helpers for the Gear Assembly DROID asset."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

_DROID_FINGERTIP_COLLISION_BOUNDS = {
    "left_inner_finger": ((0.1111, 0.0425, -0.011), (0.1491, 0.06166, 0.011)),
    "right_inner_finger": ((0.1111, -0.06166, -0.011), (0.1491, -0.0425, 0.011)),
}


def ensure_newton_compatible_droid_usd(
    usd_path: str,
    min_mass: float = 0.02,
    min_diagonal_inertia: float = 1.0e-5,
    gravity_compensation: float = 1.0,
) -> str:
    """Return a cached Droid USD suitable for Newton's MuJoCo solver."""
    from isaaclab.utils.assets import retrieve_file_path

    source = Path(retrieve_file_path(usd_path, force_download=False))
    assert source.is_file(), f"USD path must resolve to a local file: {usd_path}"

    target = source.with_name(f"{source.stem}_newton_droid{source.suffix}")
    if (
        target.exists()
        and target.stat().st_mtime >= source.stat().st_mtime
        and _is_newton_compatible(target, gravity_compensation)
    ):
        return str(target)

    shutil.copy2(source, target)
    _author_minimum_rigid_body_inertias(target, min_mass=min_mass, min_diagonal_inertia=min_diagonal_inertia)
    _author_gravity_compensation(target, gravity_compensation)
    _author_collision_mesh_leaves(target)
    _author_droid_fingertip_collisions(target)
    return str(target)


def _is_newton_compatible(usd_path: Path, gravity_compensation: float) -> bool:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"
    if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) and _needs_minimum_inertia(prim) for prim in stage.Traverse()):
        return False
    if any(
        prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetAttribute("mjc:gravcomp").Get() != gravity_compensation
        for prim in stage.Traverse()
    ):
        return False
    mimic_references = [
        relationship
        for prim in stage.Traverse()
        if "Robotiq_2F_85" in str(prim.GetPath())
        for relationship in prim.GetRelationships()
        if relationship.GetName().endswith(":referenceJoint")
    ]
    if len(mimic_references) != 5 or any(not relationship.GetTargets() for relationship in mimic_references):
        return False
    droid_collision_meshes = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and "Robotiq_2F_85" in str(prim.GetPath())
    ]
    if not _has_expected_droid_fingertip_collisions(droid_collision_meshes):
        return False
    if any(
        prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.GetName() != "newton_pad_collision"
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        for prim in droid_collision_meshes
    ):
        return False
    return all(
        prim.IsA(UsdGeom.Boundable) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _has_expected_droid_fingertip_collisions(collision_meshes) -> bool:
    from pxr import UsdGeom

    if sum(prim.GetName() != "newton_pad_collision" for prim in collision_meshes) != 11:
        return False
    proxies = {
        prim.GetParent().GetName(): prim for prim in collision_meshes if prim.GetName() == "newton_pad_collision"
    }
    if set(proxies) != set(_DROID_FINGERTIP_COLLISION_BOUNDS):
        return False
    for body_name, (expected_minimum, expected_maximum) in _DROID_FINGERTIP_COLLISION_BOUNDS.items():
        proxy = UsdGeom.Mesh(proxies[body_name])
        if proxy.ComputePurpose() != UsdGeom.Tokens.guide:
            return False
        points = proxy.GetPointsAttr().Get()
        minimum = tuple(min(float(point[axis]) for point in points) for axis in range(3))
        maximum = tuple(max(float(point[axis]) for point in points) for axis in range(3))
        if any(
            not math.isclose(actual, expected, abs_tol=1.0e-6)
            for actual, expected in zip((*minimum, *maximum), (*expected_minimum, *expected_maximum), strict=True)
        ):
            return False
    return True


def _author_minimum_rigid_body_inertias(
    usd_path: Path,
    min_mass: float,
    min_diagonal_inertia: float,
) -> None:
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    diagonal_inertia = Gf.Vec3f(min_diagonal_inertia, min_diagonal_inertia, min_diagonal_inertia)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not _needs_minimum_inertia(prim):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        if _invalid_positive_value(mass_api.GetMassAttr().Get()):
            mass_api.CreateMassAttr().Set(min_mass)
        if _invalid_diagonal_inertia(mass_api.GetDiagonalInertiaAttr().Get()):
            mass_api.CreateDiagonalInertiaAttr().Set(diagonal_inertia)
        if _invalid_center_of_mass(mass_api.GetCenterOfMassAttr().Get()):
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0))
        if _invalid_principal_axes(mass_api.GetPrincipalAxesAttr().Get()):
            mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf.GetIdentity())

    stage.GetRootLayer().Save()


def _author_gravity_compensation(usd_path: Path, gravity_compensation: float) -> None:
    """Compensate articulation gravity without disabling gravity for task objects."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(gravity_compensation)
    stage.GetRootLayer().Save()


def _author_collision_mesh_leaves(usd_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    collision_root_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Boundable)
    ]
    for path in collision_root_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsInstance():
            prim.SetInstanceable(False)
    stage.GetRootLayer().Save()
    stage.Reload()

    for path in collision_root_paths:
        root = stage.GetPrimAtPath(path)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if "Robotiq_2F_85" in str(prim.GetPath()):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                continue
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
        UsdPhysics.CollisionAPI(root).CreateCollisionEnabledAttr().Set(False)

    stage.GetRootLayer().Save()


def _author_droid_fingertip_collisions(usd_path: Path) -> None:
    """Author link-local convex pads that Newton imports without nested-transform drift."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    for body_name, (minimum, maximum) in _DROID_FINGERTIP_COLLISION_BOUNDS.items():
        body = next(
            prim for prim in stage.Traverse() if prim.GetName() == body_name and "Robotiq_2F_85" in str(prim.GetPath())
        )
        mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("newton_pad_collision"))
        points = [
            Gf.Vec3f(x, y, z)
            for z in (minimum[2], maximum[2])
            for x, y in (
                (minimum[0], minimum[1]),
                (maximum[0], minimum[1]),
                (maximum[0], maximum[1]),
                (minimum[0], maximum[1]),
            )
        ]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4] * 6)
        mesh.CreateFaceVertexIndicesAttr().Set([3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
        mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")

    stage.GetRootLayer().Save()


def _needs_minimum_inertia(prim) -> bool:
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.MassAPI):
        return True
    mass_api = UsdPhysics.MassAPI(prim)
    return (
        _invalid_positive_value(mass_api.GetMassAttr().Get())
        or _invalid_diagonal_inertia(mass_api.GetDiagonalInertiaAttr().Get())
        or _invalid_center_of_mass(mass_api.GetCenterOfMassAttr().Get())
        or _invalid_principal_axes(mass_api.GetPrincipalAxesAttr().Get())
    )


def _invalid_positive_value(value) -> bool:
    return value is None or float(value) <= 0.0


def _invalid_diagonal_inertia(value) -> bool:
    return value is None or any(float(component) <= 0.0 for component in value)


def _invalid_center_of_mass(value) -> bool:
    return value is None or any(not math.isfinite(float(component)) for component in value)


def _invalid_principal_axes(value) -> bool:
    if value is None:
        return True
    components = (value.GetReal(), *value.GetImaginary())
    return any(not math.isfinite(float(component)) for component in components) or math.isclose(
        sum(float(component) ** 2 for component in components), 0.0
    )
