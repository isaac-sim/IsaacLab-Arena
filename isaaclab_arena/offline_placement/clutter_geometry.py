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


def assert_flat_support_surface(scene: InteractiveScene, scene_key: str, spread: float) -> None:
    """Require a flat collision surface beneath the configured clutter release region.

    Args:
        scene: Scene containing the spawned support.
        scene_key: Support's runtime scene key.
        spread: Largest ClutterOn spread on this support. Edge margins are ignored
            conservatively; settled objects may still use the full support bounds.
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
        center_xy = (lower[:2] + upper[:2]) * 0.5
        half_size_xy = (upper[:2] - lower[:2]) * (0.5 * spread)
        lower[:2], upper[:2] = center_xy - half_size_xy, center_xy + half_size_xy
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
            f"Support {scene_key!r} needs a flat collision surface at its bounds' top height "
            f"covering the ClutterOn release region (spread={spread:g}). "
            "For containers, configure minimum_resting_heights_m for this support, or use an "
            "ObjectReference to a flat tabletop or tray floor as the ClutterOn parent."
        )


def _has_rectangular_top(mesh: trimesh.Trimesh, lower: np.ndarray, upper: np.ndarray) -> bool:
    """Whether a connected convex top facet covers the requested XY rectangle at upper Z."""
    import numpy as np
    from scipy.spatial import ConvexHull

    corners_xy = np.array([
        [lower[0], lower[1]],
        [lower[0], upper[1]],
        [upper[0], lower[1]],
        [upper[0], upper[1]],
    ])
    for faces, area in zip(mesh.facets, mesh.facets_area):
        vertices = mesh.triangles[faces].reshape(-1, 3)
        # Allow one micrometre of transform roundoff at the support's highest point.
        if not np.allclose(vertices[:, 2], upper[2], rtol=0, atol=1e-6):
            continue
        if not np.allclose(np.abs(mesh.face_normals[faces, 2]), 1.0, rtol=0, atol=1e-6):
            continue
        hull = ConvexHull(vertices[:, :2])
        # In 2D, hull.volume is area. Equality rejects holes and concave cutouts
        # that would otherwise be filled by the hull, including a container rim.
        if not np.isclose(hull.volume, area, rtol=1e-5, atol=1e-8):
            continue
        distances = corners_xy @ hull.equations[:, :2].T + hull.equations[:, 2]
        if np.all(distances <= 1e-6):
            return True
    return False


def prim_geometry_is_fixed(prim: Usd.Prim) -> bool:
    """Return whether geometry, descendants and ancestors have no enabled dynamic rigid body.

    Args:
        prim: Spawned prim whose support geometry is being queried.

    Returns:
        True for static collision geometry and kinematic bodies, including nested references.
    """
    from pxr import Usd

    from isaaclab_arena.utils.usd.helpers import is_enabled_dynamic_rigid_body

    assert prim.IsValid(), "Cannot inspect an invalid support prim"
    candidates = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    ancestor = prim.GetParent()
    while ancestor.IsValid() and not ancestor.IsPseudoRoot():
        candidates.append(ancestor)
        ancestor = ancestor.GetParent()
    return not any(is_enabled_dynamic_rigid_body(candidate) for candidate in candidates)


def spawned_geometry_is_fixed(scene: InteractiveScene, scene_key: str) -> bool:
    """Check support mobility from spawned physics properties for every asset variant."""
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_geometry_prim_groups

    return all(prim_geometry_is_fixed(prim) for prim, _ in get_representative_geometry_prim_groups(scene, scene_key))


def spawned_rigid_body_is_dynamic(scene: InteractiveScene, scene_key: str) -> bool:
    """Whether every spawned variant has an enabled, non-kinematic rigid body."""
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_rigid_body_prims
    from isaaclab_arena.utils.usd.helpers import is_enabled_dynamic_rigid_body

    return all(is_enabled_dynamic_rigid_body(prim) for prim in get_representative_rigid_body_prims(scene, scene_key))


def spawned_rigid_body_has_gravity(scene: InteractiveScene, scene_key: str) -> bool:
    """Whether all variants of a spawned rigid object participate in gravity."""
    from isaaclab_arena.environments.arena_world_scene_access import get_representative_rigid_body_prims

    # Both backends import this USD attribute despite its PhysX namespace.
    # It describes authored gravity intent; Newton does not support per-body gravity exclusion.
    return all(
        body.GetAttribute("physxRigidBody:disableGravity").Get() is not True
        for body in get_representative_rigid_body_prims(scene, scene_key)
    )
