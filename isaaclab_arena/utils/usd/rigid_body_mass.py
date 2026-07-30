# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read and combine what the rigid bodies in a USD asset weigh and how they resist spinning."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from pxr import Gf, Usd, UsdPhysics


@dataclass(frozen=True)
class RigidBodyMass:
    """What a rigid body weighs and how it resists spinning."""

    mass: float
    """Mass in kilograms."""

    center_of_mass: tuple[float, float, float]
    """Centre of mass, in the body's own frame."""

    diagonal_inertia: tuple[float, float, float]
    """Moments of inertia about the principal axes."""

    principal_axes: tuple[float, float, float, float]
    """Rotation from the principal axes to the body's frame, as ``(x, y, z, w)``."""


def read_rigid_body_mass(prim: Usd.Prim) -> RigidBodyMass | None:
    """Read what a rigid body weighs, or None if the asset did not spell it all out.

    Args:
        prim: Prim to read from.

    Returns:
        The rigid body's mass, or None if mass, centre of mass, or inertia is missing.
    """
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return None
    api = UsdPhysics.MassAPI(prim)
    mass_attribute = api.GetMassAttr()
    center_attribute = api.GetCenterOfMassAttr()
    inertia_attribute = api.GetDiagonalInertiaAttr()
    if not all(attribute.HasAuthoredValue() for attribute in (mass_attribute, center_attribute, inertia_attribute)):
        return None
    axes = api.GetPrincipalAxesAttr().Get() or Gf.Quatf(1.0)
    imaginary = axes.GetImaginary()
    return RigidBodyMass(
        mass=float(mass_attribute.Get()),
        center_of_mass=tuple(float(value) for value in center_attribute.Get()),
        diagonal_inertia=tuple(float(value) for value in inertia_attribute.Get()),
        # x, y, z, w
        principal_axes=tuple(float(value) for value in imaginary) + (float(axes.GetReal()),),
    )


def write_rigid_body_mass(prim: Usd.Prim, body_mass: RigidBodyMass) -> None:
    """Write what a rigid body weighs onto a prim, overriding anything its shapes would have implied.

    Args:
        prim: Prim to write to.
        body_mass: What the body weighs and how it resists spinning.
    """
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr().Set(float(body_mass.mass))
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*body_mass.center_of_mass))
    api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*body_mass.diagonal_inertia))
    *imaginary, real = body_mass.principal_axes
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(real, Gf.Vec3f(*imaginary)))


def _quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Turn an ``(x, y, z, w)`` quaternion into a rotation matrix."""
    x, y, z, real = quaternion
    # Gf stores matrices the other way round from numpy, so transpose what it hands back.
    return np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(real, Gf.Vec3d(x, y, z))))).T


def _matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Turn a rotation matrix into an ``(x, y, z, w)`` quaternion."""
    # Gf stores matrices the other way round from numpy, so hand it the transpose.
    gf_matrix = Gf.Matrix4d(1.0)
    gf_matrix.SetRotateOnly(Gf.Matrix3d(*matrix.T.flatten().tolist()))
    quaternion = gf_matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return tuple(float(value) for value in imaginary) + (float(quaternion.GetReal()),)


def _to_principal_axes(inertia: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """Split an inertia tensor into moments about its principal axes and their rotation."""
    moments, axes = np.linalg.eigh(inertia)
    # eigh can hand back a mirrored frame, which is not a rotation. Flip one axis to fix it.
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    return tuple(float(value) for value in moments), _matrix_to_quaternion(axes)


def combine_rigid_body_masses(parts: list[tuple[RigidBodyMass, Gf.Matrix4d]]) -> RigidBodyMass:
    """Work out what several parts weigh and how they spin once treated as one solid piece.

    Args:
        parts: What each part weighs, and the transform from the part's frame into the merged
            body's frame.

    Returns:
        What the parts weigh taken together.
    """
    assert parts, "cannot combine the mass of nothing"
    total_mass = sum(body_mass.mass for body_mass, _ in parts)
    assert total_mass > 0.0, f"the parts add up to a mass of {total_mass}"

    centers = []
    rotations = []
    for body_mass, transform in parts:
        point = transform.Transform(Gf.Vec3d(*body_mass.center_of_mass))
        centers.append(np.array([point[0], point[1], point[2]]))
        rotation = np.array(transform.ExtractRotationMatrix()).T
        rotations.append(rotation @ _quaternion_to_matrix(body_mass.principal_axes))

    center_of_mass = sum(body_mass.mass * center for (body_mass, _), center in zip(parts, centers)) / total_mass

    inertia = np.zeros((3, 3))
    for (body_mass, _), center, rotation in zip(parts, centers, rotations):
        part_inertia = rotation @ np.diag(body_mass.diagonal_inertia) @ rotation.T
        # Move each part's inertia to the shared centre of mass before adding it in.
        offset = center - center_of_mass
        inertia += part_inertia + body_mass.mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))

    moments, axes = _to_principal_axes(inertia)
    return RigidBodyMass(
        mass=float(total_mass),
        center_of_mass=tuple(float(value) for value in center_of_mass),
        diagonal_inertia=moments,
        principal_axes=axes,
    )
