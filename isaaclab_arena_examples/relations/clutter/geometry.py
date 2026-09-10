# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Support geometry and spawned physics checks for offline clutter generation."""

from __future__ import annotations

import math

from isaaclab.scene import InteractiveScene
from pxr import Usd, UsdPhysics

from isaaclab_arena.environments.arena_world_scene_access import (
    _find_single_rigid_body_prim_in_subtree,
    _get_representative_prim_groups,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena_examples.relations.clutter.drop_poses import ClutterRegion

_QUARTER_TURN_TOLERANCE_RAD = 1e-3


def region_above_support(
    support_position: tuple[float, float, float],
    support_bbox: AxisAlignedBoundingBox,
    spread: float = 1.0,
    env_index: int = 0,
    support_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> ClutterRegion:
    """Return the top-face region of a horizontal, axis-aligned support.

    Args:
        support_position: Support position in environment frame E, shape (3,).
        support_bbox: Object-local bounds, min/max shape (N, 3); N is the environment count.
        spread: Usable fraction of the support footprint, in (0, 1].
        env_index: Environment row in support_bbox.
        support_rotation_xyzw: Support-to-E quaternion, shape (4,); yaw must be a quarter turn.
    """
    quarters = quaternion_to_90_deg_z_quarters(support_rotation_xyzw, tol_deg=math.degrees(_QUARTER_TURN_TOLERANCE_RAD))
    bounds = support_bbox.rotated_90_around_z(quarters)
    lower, upper = bounds.min_point[env_index], bounds.max_point[env_index]
    region = ClutterRegion(
        min_x=float(lower[0]) + support_position[0],
        min_y=float(lower[1]) + support_position[1],
        max_x=float(upper[0]) + support_position[0],
        max_y=float(upper[1]) + support_position[1],
        floor_z=float(upper[2]) + support_position[2],
    )
    return region.scaled(spread) if spread != 1.0 else region


def prim_geometry_is_fixed(prim: Usd.Prim) -> bool:
    """Return whether neither geometry nor its ancestors have an enabled dynamic rigid body.

    Args:
        prim: Spawned prim whose support geometry is being queried.

    Returns:
        True for static collision geometry and kinematic bodies, including nested references.
    """
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
    path = getattr(scene.cfg, scene_key).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
    return all(prim_geometry_is_fixed(prim) for prim, _ in _get_representative_prim_groups(scene, scene_key, path))


def spawned_rigid_body_has_gravity(scene: InteractiveScene, scene_key: str) -> bool:
    """Whether all variants of a spawned rigid object participate in gravity."""
    assert scene_key in scene.rigid_objects, f"Scene key {scene_key!r} is not a rigid object"
    path = getattr(scene.cfg, scene_key).prim_path.format(ENV_REGEX_NS=scene.env_regex_ns)
    for prim, _ in _get_representative_prim_groups(scene, scene_key, path):
        body = _find_single_rigid_body_prim_in_subtree(prim, scene_key)
        # Isaac Lab's solver-common RigidBodyBaseCfg maps disable_gravity to this USD attribute.
        if body.GetAttribute("physxRigidBody:disableGravity").Get() is True:
            return False
    return True


def resting_extents(
    bbox: AxisAlignedBoundingBox, rotation_xyzw: tuple[float, float, float, float]
) -> tuple[float, float, float, float, float]:
    """Return rotated (min_x, min_y, max_x, max_y, min_z) offsets from the object origin."""
    rotated = bbox.rotated_by_quat(rotation_xyzw)
    lower, upper = rotated.min_point[0], rotated.max_point[0]
    return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]), float(lower[2])


def dynamic_rigid_object_keys(scene: InteractiveScene) -> list[str]:
    """Return scene keys whose rigid geometry can move under physics."""
    return [key for key in scene.rigid_objects if not spawned_geometry_is_fixed(scene, key)]
