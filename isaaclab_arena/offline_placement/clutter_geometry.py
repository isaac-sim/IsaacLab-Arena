# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Support geometry and spawned physics checks for offline clutter generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.utils.physics_settle import get_pose_drift

if TYPE_CHECKING:
    import numpy as np
    import torch
    import trimesh

    from isaaclab.scene import InteractiveScene
    from pxr import Usd


def fixed_poses_match(expected: torch.Tensor, current: torch.Tensor) -> bool:
    """Compare fixed xyz/xyzw poses (..., 7) in the same frame, allowing float32 roundoff."""
    drift = get_pose_drift(expected, current)
    return drift is not None and drift[0] <= 1e-5 and drift[1] <= 1e-3


def assert_support_reference_transform(scene: InteractiveScene, scene_key: str) -> None:
    """Require reference transforms that can be read without changing live collider operations."""
    if scene_key not in scene.extras:
        return

    import isaaclab.sim as sim_utils

    from isaaclab_arena.environments.arena_world_scene_access import get_representative_geometry_prim_groups

    for root, _ in get_representative_geometry_prim_groups(scene, scene_key):
        assert sim_utils.validate_standard_xform_ops(root), (
            f"Support reference {scene_key!r} needs translate, orient and scale operations authored "
            "before scene construction; changing collider transforms after spawning invalidates physics views."
        )


def assert_flat_support_surface(scene: InteractiveScene, scene_key: str) -> None:
    """Require the support bounds' top face to be covered by a flat collision surface.

    Used for supports without an explicit minimum resting height. Check each
    spawned geometry group before collecting any layouts.
    """
    import numpy as np
    import trimesh

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.environments.arena_world_scene_access import get_representative_geometry_prim_groups
    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for root, _ in get_representative_geometry_prim_groups(scene, scene_key):
        bounds = bounds_cache.ComputeWorldBound(root).ComputeAlignedRange()
        lower, upper = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        has_surface = False
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            if prim.IsA(UsdGeom.Cube):
                size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                mesh = trimesh.creation.box(extents=(size, size, size))
            elif prim.IsA(UsdGeom.Mesh):
                mesh = extract_trimesh_from_prim(prim.GetStage(), str(prim.GetPath()))
            else:
                continue
            # Map geometry frame G into simulation world W; USD uses row vectors, trimesh column vectors.
            T_W_G = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            mesh.apply_transform(T_W_G)
            if _has_rectangular_top(mesh, lower, upper):
                has_surface = True
                break
        assert has_surface, (
            f"Support {scene_key!r} needs a flat rectangular collision surface covering its bounds' top face. "
            "For containers, configure minimum_resting_heights_m for this support, or use an "
            "ObjectReference to a flat tabletop or tray floor as the ClutterOn parent."
        )


def _has_rectangular_top(mesh: trimesh.Trimesh, lower: np.ndarray, upper: np.ndarray) -> bool:
    """Whether one connected planar facet covers the bounds' full rectangular top."""
    import numpy as np

    footprint_area = float(np.prod((upper - lower)[:2]))
    for faces, area in zip(mesh.facets, mesh.facets_area):
        vertices = mesh.triangles[faces].reshape(-1, 3)
        # Allow one micrometre of transform roundoff when comparing surface coordinates.
        at_top = np.allclose(vertices[:, 2], upper[2], rtol=0, atol=1e-6)
        covers_footprint = np.allclose(
            [vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)],
            [lower[:2], upper[:2]],
            rtol=0,
            atol=1e-6,
        )
        if at_top and covers_footprint and np.isclose(area, footprint_area, rtol=1e-5, atol=1e-8):
            return True
    return False


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

    # Both backends import this USD attribute despite its PhysX namespace.
    # It describes authored gravity intent; Newton does not support per-body gravity exclusion.
    return all(
        body.GetAttribute("physxRigidBody:disableGravity").Get() is not True
        for body in get_representative_rigid_body_prims(scene, scene_key)
    )
