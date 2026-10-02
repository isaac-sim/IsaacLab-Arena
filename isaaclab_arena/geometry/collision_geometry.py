# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read configured rigid-asset collision bounds and measure their sampled separation."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from isaaclab.utils.math import matrix_from_quat


@dataclass(frozen=True)
class CollisionBoxes:
    """Represent conservative collision boxes in one shared coordinate frame."""

    centers: torch.Tensor
    """Box centers, with shape (N, 3)."""

    axes: torch.Tensor
    """Orthonormal box axes as rows, with shape (N, 3, 3)."""

    half_extents: torch.Tensor
    """Positive half sizes along the corresponding axes, with shape (N, 3)."""

    features: tuple[str, ...]
    """Optional authored feature labels, or collider paths when no label is present."""

    def __post_init__(self) -> None:
        count = len(self.features)
        assert self.centers.shape == self.half_extents.shape == (count, 3)
        assert self.axes.shape == (count, 3, 3)
        assert self.centers.device == self.axes.device == self.half_extents.device
        assert self.centers.dtype == self.axes.dtype == self.half_extents.dtype


@dataclass(frozen=True)
class CollisionPrimitives:
    """Retain primitive shape information alongside conservative collision boxes."""

    boxes: CollisionBoxes
    cylinder_axes: torch.Tensor
    """Local axial coordinate for each cylinder, or -1 for a box, shape (N,)."""

    def __post_init__(self) -> None:
        assert self.cylinder_axes.shape == (len(self.boxes.features),)
        assert self.cylinder_axes.device == self.boxes.centers.device
        assert self.cylinder_axes.dtype == torch.int64
        assert bool(((self.cylinder_axes >= -1) & (self.cylinder_axes <= 2)).all())

    def bounds_in_frame(self, T_P_O: torch.Tensor) -> torch.Tensor:
        """Project the solid primitives onto a parent frame's three axes.

        Args:
            T_P_O: Object-to-parent XYZ+XYZW poses, shape (..., 7).

        Returns:
            Exact union bounds for boxes and circular cylinders, shape (..., 2, 3).
            Cylinder support uses its curved surface, excluding empty bounding-box corners.
        """
        assert T_P_O.shape[-1] == 7
        R_P_O = matrix_from_quat(T_P_O[..., 3:])
        centers = torch.einsum("...ij,nj->...ni", R_P_O, self.boxes.centers) + T_P_O[..., None, :3]
        axes = torch.einsum("...ij,nkj->...nki", R_P_O, self.boxes.axes)
        projections = axes * self.boxes.half_extents[..., None]
        box_radius = projections.abs().sum(dim=-2)
        axial = torch.arange(3, device=self.cylinder_axes.device) == self.cylinder_axes[:, None]
        cylinder_radius = (projections.abs() * axial[..., None]).sum(dim=-2)
        cylinder_radius += (projections.square() * (~axial)[..., None]).sum(dim=-2).sqrt()
        radii = torch.where(self.cylinder_axes[:, None] >= 0, cylinder_radius, box_radius)
        return torch.stack(((centers - radii).amin(dim=-2), (centers + radii).amax(dim=-2)), dim=-2)


def read_collision_boxes(
    usd_path: str | Path,
    *,
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CollisionBoxes:
    """Return conservative boxes for the configured static box/cylinder geometry.

    Args:
        usd_path: Composed nonarticulated spawn USD with a rigid default-prim frame.
        scale: Positive spawn scale in the asset-root frame.
        device: Device for cached geometry tensors.
        dtype: Floating dtype for cached geometry tensors.

    Returns:
        Conservative bounds from the validated primitive reader.
    """
    return read_collision_primitives(usd_path, scale=scale, device=device, dtype=dtype).boxes


def read_collision_primitives(
    usd_path: str | Path,
    *,
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CollisionPrimitives:
    """Read enabled cube/cylinder colliders from one configured, nonarticulated USD asset.

    Args:
        usd_path: Configured spawn USD, including its composed physics layer and references.
        scale: Positive spawn scale applied in the asset-root frame.
        device: Device for the cached collision tensors.
        dtype: Floating dtype for the cached collision tensors.

    Returns:
        Primitive shapes and bounds in the default prim's local frame, in meters.
        Invisible colliders of any rendering purpose are included. Physical
        schema dimensions are authoritative; authored display extents are ignored.
        Animated geometry, meshes, articulations, moving child bodies, nonrigid
        root transforms, and sheared boxes require other geometry representations.
        Read once during environment geometry setup and reuse the result.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    assert len(scale) == 3 and all(math.isfinite(value) and value > 0 for value in scale)
    stage = Usd.Stage.Open(str(usd_path))
    assert stage, f"Cannot read configured collision asset: {usd_path}"
    assert math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0), "Collision bounds require meter-based USD assets."
    root = stage.GetDefaultPrim()
    assert root.IsValid(), "A collision asset must have a default prim."
    ancestor = root.GetParent()
    while ancestor:
        xform = UsdGeom.Xformable(ancestor)
        if xform:
            assert not (
                xform.GetXformOpOrderAttr().GetNumTimeSamples()
                or any(operation.GetAttr().GetNumTimeSamples() for operation in xform.GetOrderedXformOps())
            ), "Animated frame ancestry cannot use cached collision bounds."
        ancestor = ancestor.GetParent()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_linear = cache.GetLocalToWorldTransform(root).ExtractRotationMatrix()
    root_product = root_linear * root_linear.GetTranspose()
    assert Gf.IsClose(root_product, Gf.Matrix3d(1.0), 1e-6) and math.isclose(
        root_linear.GetDeterminant(), 1.0, abs_tol=1e-6
    ), "Authored root scale, shear, or reflection cannot be represented by a measured rigid pose."
    scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))
    centers, axes, half_extents, features, cylinder_axes = [], [], [], [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        assert not prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ), "Read articulated collision geometry one link at a time."
        assert not prim.IsA(UsdPhysics.Joint), "Jointed assets require live link transforms."
        xform = UsdGeom.Xformable(prim)
        if xform:
            assert not (
                xform.GetXformOpOrderAttr().GetNumTimeSamples()
                or any(operation.GetAttr().GetNumTimeSamples() for operation in xform.GetOrderedXformOps())
            ), "Animated transforms cannot use cached collision bounds."
            assert prim == root or not xform.GetResetXformStack(), "Reset child transforms require a separate frame."
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()
            assert not enabled.GetNumTimeSamples(), "Animated body enablement cannot use cached collision bounds."
            if prim != root:
                assert not enabled.Get(), f"Moving child collision body requires its own frame: {prim.GetPath()}"
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        assert not enabled.GetNumTimeSamples(), "Animated collider enablement cannot use cached collision bounds."
        if not enabled.Get():
            continue
        feature = prim.GetCustomDataByKey("arena:feature")
        if feature is None:
            feature = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        assert isinstance(feature, str) and feature, f"Invalid collision feature label: {prim.GetPath()}"
        dimensions = _physical_dimensions(prim)
        # USD uses row-vector matrices. Spawn scale acts after the primitive's
        # transform into the asset-root frame, including its local translation.
        relative, _ = cache.ComputeRelativeTransform(prim, root)
        matrix = relative * scale_matrix
        center = matrix.Transform(Gf.Vec3d(0.0))
        shape_axes, shape_extents = [], []
        for index in range(3):
            basis = Gf.Vec3d(0.0)
            basis[index] = 1.0
            vector = matrix.TransformDir(basis)
            length = vector.GetLength()
            assert math.isfinite(length) and length > 0.0, f"Degenerate collider transform: {prim.GetPath()}"
            shape_axes.append(tuple(vector / length))
            shape_extents.append(dimensions[index] * length / 2)
        if prim.IsA(UsdGeom.Cylinder):
            axial = ("X", "Y", "Z").index(UsdGeom.Cylinder(prim).GetAxisAttr().Get())
            radial_scales = [2 * shape_extents[index] / dimensions[index] for index in range(3) if index != axial]
            assert math.isclose(*radial_scales, rel_tol=1e-6), "Unequal radial cylinder scale is unsupported."
            cylinder_axes.append(axial)
        else:
            cylinder_axes.append(-1)
        centers.append(tuple(center))
        axes.append(shape_axes)
        half_extents.append(shape_extents)
        features.append(feature)
    assert centers, f"Asset has no enabled collision primitives: {usd_path}"
    boxes = CollisionBoxes(
        torch.tensor(centers, device=device, dtype=dtype),
        torch.tensor(axes, device=device, dtype=dtype),
        torch.tensor(half_extents, device=device, dtype=dtype),
        tuple(features),
    )
    assert bool(torch.isfinite(boxes.centers).all() & torch.isfinite(boxes.half_extents).all())
    assert bool((boxes.half_extents > 0.0).all()), "Collision bounds require positive volume."
    identity = torch.eye(3, device=device, dtype=dtype).expand_as(boxes.axes)
    assert torch.allclose(
        boxes.axes @ boxes.axes.transpose(-1, -2), identity, atol=1e-6, rtol=0
    ), "Sheared collider bounds cannot be represented by orthogonal boxes."
    return CollisionPrimitives(boxes, torch.tensor(cylinder_axes, device=device, dtype=torch.int64))


def _physical_dimensions(prim) -> tuple[float, float, float]:
    """Derive local primitive bounds from collision-schema dimensions, ignoring display extents."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Cube):
        attributes = (UsdGeom.Cube(prim).GetSizeAttr(),)
    else:
        assert prim.IsA(UsdGeom.Cylinder), f"Only physical cube/cylinder colliders are supported: {prim.GetPath()}"
        cylinder = UsdGeom.Cylinder(prim)
        attributes = (cylinder.GetRadiusAttr(), cylinder.GetHeightAttr(), cylinder.GetAxisAttr())
    assert not any(
        attribute.GetNumTimeSamples() for attribute in attributes
    ), "Animated collision dimensions cannot use cached bounds."
    if len(attributes) == 1:
        size = float(attributes[0].Get())
        dimensions = (size, size, size)
    else:
        radius, height, axis = (attribute.Get() for attribute in attributes)
        assert axis in ("X", "Y", "Z"), "A cylinder requires a physical axis."
        dimensions = [2 * float(radius)] * 3
        dimensions[("X", "Y", "Z").index(axis)] = float(height)
        dimensions = tuple(dimensions)
    assert all(math.isfinite(value) and value > 0.0 for value in dimensions), "Collider dimensions must be positive."
    return dimensions


def transform_boxes(boxes: CollisionBoxes, T_P_O: torch.Tensor) -> CollisionBoxes:
    """Map cached object-frame collision boxes into a parent frame using one XYZ+XYZW pose.

    Args:
        boxes: Collision boxes expressed in object frame O.
        T_P_O: Pose mapping O into P, with shape (7,) and the boxes' dtype/device.

    Returns:
        New parent-frame boxes, preserving local half extents and feature names.
    """
    assert T_P_O.shape == (7,), "Transform one object's collision boxes with one XYZ+XYZW pose."
    assert bool(torch.isfinite(T_P_O).all()) and float(torch.linalg.vector_norm(T_P_O[3:])) > 1e-12
    R_P_O = matrix_from_quat(T_P_O[3:])
    return CollisionBoxes(
        boxes.centers @ R_P_O.T + T_P_O[:3],
        boxes.axes @ R_P_O.T,
        boxes.half_extents,
        boxes.features,
    )


def pairwise_box_separation(left: CollisionBoxes, right: CollisionBoxes) -> torch.Tensor:
    """Measure a signed separating-axis bound for every pair of boxes in a shared frame.

    Args:
        left: First collection of N boxes.
        right: Second collection of M boxes in the same frame and dtype/device.

    Returns:
        An (N, M) tensor. Positive values bound separation from below; zero means
        touching boxes, and negative values mean overlapping conservative boxes.
        Box overlap does not establish contact between curved primitives. These are projection gaps,
        not Euclidean distances or swept-path certificates. No contact allowance
        is applied; any task-specific contact geometry must be checked separately.
    """
    A = left.axes[:, None, :, :]
    B = right.axes[None, :, :, :]
    shape = (len(left.features), len(right.features), 3, 3)
    cross = torch.linalg.cross(A[..., :, None, :], B[..., None, :, :], dim=-1).flatten(-3, -2)
    axes = torch.cat((A.expand(shape), B.expand(shape), cross), dim=-2)
    lengths = torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    valid = lengths.squeeze(-1) > 1e-12
    axes = axes / lengths.clamp_min(1e-12)
    delta = right.centers[None, :, :] - left.centers[:, None, :]
    distance = (axes * delta[..., None, :]).sum(-1).abs()
    radius_left = ((axes @ A.transpose(-1, -2)).abs() * left.half_extents[:, None, None, :]).sum(-1)
    radius_right = ((axes @ B.transpose(-1, -2)).abs() * right.half_extents[None, :, None, :]).sum(-1)
    gaps = distance - radius_left - radius_right
    return gaps.masked_fill(~valid, -torch.inf).amax(-1)
