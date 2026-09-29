# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Support geometry and spawned physics checks for offline clutter generation."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from pxr import Usd


@dataclass(frozen=True)
class ClutterRegion:
    """Axis-aligned support footprint and surface height, in the environment frame E."""

    min_x: float
    """Minimum X in E, in metres."""

    min_y: float
    """Minimum Y in E, in metres."""

    max_x: float
    """Maximum X in E, in metres."""

    max_y: float
    """Maximum Y in E, in metres."""

    floor_z: float
    """Z of the surface objects are dropped onto."""

    def __post_init__(self) -> None:
        assert self.max_x > self.min_x, f"region needs max_x > min_x, got {self.min_x}, {self.max_x}"
        assert self.max_y > self.min_y, f"region needs max_y > min_y, got {self.min_y}, {self.max_y}"


def get_placement_region(
    support_position: tuple[float, float, float],
    placement_region_bbox: AxisAlignedBoundingBox,
    support_rotation_xyzw: tuple[float, float, float, float],
) -> ClutterRegion:
    """Return the top-face region of a horizontal, axis-aligned support.

    Args:
        support_position: Support position in environment frame E, shape (3,).
        placement_region_bbox: Object-local bounds for one environment, min/max shape (1, 3).
        support_rotation_xyzw: Support-to-E quaternion, shape (4,); yaw must be a quarter turn.
    """
    quarters = quaternion_to_90_deg_z_quarters(support_rotation_xyzw)
    bounds = placement_region_bbox.rotated_90_around_z(quarters)
    lower, upper = bounds.min_point[0], bounds.max_point[0]
    return ClutterRegion(
        min_x=float(lower[0]) + support_position[0],
        min_y=float(lower[1]) + support_position[1],
        max_x=float(upper[0]) + support_position[0],
        max_y=float(upper[1]) + support_position[1],
        floor_z=float(upper[2]) + support_position[2],
    )


def prim_geometry_is_fixed(prim: Usd.Prim) -> bool:
    """Return whether geometry, descendants and ancestors have no enabled dynamic rigid body.

    Args:
        prim: Spawned prim whose support geometry is being queried.

    Returns:
        True for static collision geometry and kinematic bodies, including nested references.
    """
    from pxr import Usd, UsdPhysics

    assert prim.IsValid(), "Cannot inspect an invalid support prim"
    candidates = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    ancestor = prim.GetParent()
    while ancestor.IsValid() and not ancestor.IsPseudoRoot():
        candidates.append(ancestor)
        ancestor = ancestor.GetParent()
    for candidate in candidates:
        if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(candidate)
            if body.GetRigidBodyEnabledAttr().Get() and not body.GetKinematicEnabledAttr().Get():
                return False
    return True


def spawned_geometry_is_fixed(scene: InteractiveScene, scene_key: str) -> bool:
    """Check support mobility from spawned physics properties for every asset variant."""
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_geometry_prim_groups

    return all(prim_geometry_is_fixed(prim) for prim, _ in get_representative_geometry_prim_groups(scene, scene_key))


def spawned_rigid_body_is_dynamic(scene: InteractiveScene, scene_key: str) -> bool:
    """Whether every spawned variant has an enabled, non-kinematic rigid body."""
    from pxr import UsdPhysics

    from isaaclab_arena.environments.arena_world_scene_access import get_representative_rigid_body_prims

    for prim in get_representative_rigid_body_prims(scene, scene_key):
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
            return False
    return True


def spawned_rigid_body_has_gravity(scene: InteractiveScene, scene_key: str) -> bool:
    """Whether all variants of a spawned rigid object participate in gravity."""
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_rigid_body_prims

    # Isaac Lab's solver-common RigidBodyBaseCfg maps disable_gravity to this USD attribute.
    return all(
        body.GetAttribute("physxRigidBody:disableGravity").Get() is not True
        for body in get_representative_rigid_body_prims(scene, scene_key)
    )


@dataclass
class ClutterContainmentResult:
    """Indices of clutter members that failed containment checks."""

    diverged: list[int] = field(default_factory=list)
    """Indices of non-finite poses."""

    fell_through: list[int] = field(default_factory=list)
    """Indices below the support surface."""

    fell_off: list[int] = field(default_factory=list)
    """Indices outside the support footprint."""

    @property
    def ok(self) -> bool:
        """Whether every member satisfies the containment checks."""
        return not (self.diverged or self.fell_through or self.fell_off)

    def describe(self, names: list[str]) -> str:
        """Return a human-readable summary naming the offending members."""
        parts = []
        for label, indices in (
            ("diverged", self.diverged),
            ("fell through", self.fell_through),
            ("fell off", self.fell_off),
        ):
            if indices:
                offenders = ", ".join(names[index] for index in indices)
                parts.append(f"{label}: {offenders}")
        return "; ".join(parts) if parts else "all members within support"


def check_resting_poses(
    bounds: AxisAlignedBoundingBox,
    region: ClutterRegion,
    containment_margin_m: float,
    fall_through_tolerance_m: float,
) -> ClutterContainmentResult:
    """Return containment failures for N members.

    Args:
        bounds: Rotated object bounds in the environment frame, min/max shape (N, 3).
        region: Full support footprint and surface height, without the release spread scaling.
        containment_margin_m: Permitted overhang beyond the support footprint.
        fall_through_tolerance_m: Permitted penetration below the support surface.
    """
    verdict = ClutterContainmentResult()
    margin = containment_margin_m
    floor = region.floor_z - fall_through_tolerance_m
    for index, (lower, upper) in enumerate(zip(bounds.min_point, bounds.max_point, strict=True)):
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            verdict.diverged.append(index)
            continue
        if lower[2] < floor:
            verdict.fell_through.append(index)
        if not (
            lower[0] >= region.min_x - margin
            and upper[0] <= region.max_x + margin
            and lower[1] >= region.min_y - margin
            and upper[1] <= region.max_y + margin
        ):
            verdict.fell_off.append(index)
    return verdict
