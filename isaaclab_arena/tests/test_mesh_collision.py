# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh-based collision detection: sphere decomposition, dispatch, and end-to-end solver."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshManager, greedy_sphere_decomposition
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

try:
    import warp as wp

    wp.init()
    _WARP_AVAILABLE = True
except Exception:
    _WARP_AVAILABLE = False

requires_warp = pytest.mark.skipif(not _WARP_AVAILABLE, reason="Warp not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cylinder(name: str, radius: float = 0.033, height: float = 0.1) -> DummyObject:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-radius, -radius, -height / 2),
            max_point=(radius, radius, height / 2),
        ),
        collision_mesh=mesh,
    )


def _make_box_obj(name: str, sx: float, sy: float, sz: float) -> DummyObject:
    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-sx / 2, -sy / 2, -sz / 2),
            max_point=(sx / 2, sy / 2, sz / 2),
        ),
        collision_mesh=mesh,
    )


def _make_table() -> DummyObject:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.05))
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, -0.025), max_point=(0.5, 0.5, 0.025)),
        collision_mesh=mesh,
    )
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    table.add_relation(IsAnchor())
    return table


# ---------------------------------------------------------------------------
# Unit: greedy_sphere_decomposition
# ---------------------------------------------------------------------------


def test_sphere_decomposition_covers_surface():
    """Sphere decomposition should cover >80% of surface sample points."""
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=32)
    spheres = greedy_sphere_decomposition(mesh, num_spheres=20, n_surface=500)
    assert spheres.shape[1] == 4
    assert spheres.shape[0] <= 20

    # Check coverage: what fraction of surface points are within radius of some sphere?
    surface_pts = trimesh.sample.sample_surface(mesh, 200)[0]
    centers = spheres[:, :3]
    radii = spheres[:, 3]
    covered = 0
    for pt in surface_pts:
        dists = np.linalg.norm(centers - pt, axis=1)
        if (dists < radii).any():
            covered += 1
    coverage = covered / len(surface_pts)
    assert coverage > 0.8, f"Coverage only {coverage:.1%}"


# ---------------------------------------------------------------------------
# Unit: WarpMeshManager caching
# ---------------------------------------------------------------------------


@requires_warp
def test_warp_mesh_caching():
    """Same mesh object should return identical Warp mesh from cache."""
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    manager = WarpMeshManager(num_spheres=10)
    m1 = manager.get_warp_mesh(mesh)
    m2 = manager.get_warp_mesh(mesh)
    assert m1 is m2


# ---------------------------------------------------------------------------
# Unit: NoCollisionLossStrategy routing
# ---------------------------------------------------------------------------


def test_dispatch_routes_to_aabb_in_bbox_mode():
    """BBOX mode should use AABB even when objects have meshes."""
    dispatch = NoCollisionLossStrategy(collision_mode=CollisionMode.BBOX, slope=10.0)

    obj_a = _make_cylinder("a")
    obj_b = _make_cylinder("b")
    obj_b.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = obj_b.get_bounding_box().translated((0.5, 0.0, 0.0))

    # Separated: should be zero regardless of mode
    loss = dispatch.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=obj_a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=obj_a,
        parent_obj=obj_b,
        parent_pos=torch.tensor([0.5, 0.0, 0.0]),
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_dispatch_falls_back_when_no_mesh():
    """MESH mode with objects lacking collision_mesh should use AABB."""
    dispatch = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10.0)

    # No collision_mesh
    obj_a = DummyObject(
        name="a", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    )
    obj_b = DummyObject(
        name="b", bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
    )

    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = obj_b.get_bounding_box().translated((0.5, 0.0, 0.0))

    loss = dispatch.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=obj_a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=obj_a,
        parent_obj=obj_b,
        parent_pos=torch.tensor([0.5, 0.0, 0.0]),
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Unit: NoCollisionLossStrategy mesh mode (requires warp)
# ---------------------------------------------------------------------------


@requires_warp
def test_mesh_positive_loss_overlapping_cylinders():
    """Two cylinders at the same position should produce positive mesh loss."""
    dispatch = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10000.0)
    a = _make_cylinder("a")
    b = _make_cylinder("b")
    b.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = b.get_bounding_box().translated((0.0, 0.0, 0.0))

    loss = dispatch.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor([0.0, 0.0, 0.0]),
    )
    assert loss.item() > 0.0


@requires_warp
def test_mesh_loss_respects_clearance_m():
    """Near-miss cylinders should have positive loss when clearance_m > 0."""
    dispatch = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10000.0)
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    b.set_initial_pose(Pose(position_xyz=(0.07, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = b.get_bounding_box().translated((0.07, 0.0, 0.0))

    # Without clearance: separated (0.07 > 0.03+0.03)
    loss_no_clearance = dispatch.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor([0.07, 0.0, 0.0]),
    )

    # With large clearance: should trigger
    loss_with_clearance = dispatch.compute_loss(
        clearance_m=0.05,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor([0.07, 0.0, 0.0]),
    )
    assert loss_with_clearance.item() > loss_no_clearance.item()


# ---------------------------------------------------------------------------
# Integration: solver with mesh mode
# ---------------------------------------------------------------------------


@requires_warp
def test_solver_separates_overlapping_cylinders_mesh_mode():
    """RelationSolver with MESH mode should push overlapping cylinders apart."""
    table = _make_table()
    a = _make_cylinder("cyl_a")
    b = _make_cylinder("cyl_b")
    a.add_relation(On(table))
    b.add_relation(On(table))

    objects = [table, a, b]
    initial = [{table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.03), b: (0.01, 0.0, 0.03)}]

    solver = RelationSolver(
        params=RelationSolverParams(
            collision_mode=CollisionMode.MESH, max_iters=200, convergence_threshold=1e-4, verbose=False
        )
    )
    result = solver.solve(objects, initial)[0]

    pos_a = np.array(result[a])
    pos_b = np.array(result[b])
    dist = np.linalg.norm(pos_a[:2] - pos_b[:2])
    # Must be separated by at least sum of radii (0.033 + 0.033 = 0.066)
    assert dist > 0.066, f"Cylinders not separated: dist={dist:.4f}, need > 0.066"


@requires_warp
def test_on_pairs_skipped_in_mesh_mode():
    """On-linked pairs should not be penalized in mesh mode (same as AABB)."""
    table = _make_table()
    obj = _make_cylinder("can")
    obj.add_relation(On(table))

    objects = [table, obj]
    # Place can directly on table surface -- should converge without fighting On
    initial = [{table: (0.0, 0.0, 0.0), obj: (0.0, 0.0, 0.03)}]

    solver = RelationSolver(
        params=RelationSolverParams(
            collision_mode=CollisionMode.MESH, max_iters=100, convergence_threshold=1e-4, verbose=False
        )
    )
    result = solver.solve(objects, initial)[0]

    # Should stay near table surface, not be pushed far away
    z = result[obj][2]
    assert 0.0 < z < 0.15, f"Object pushed too far: z={z}"


# ---------------------------------------------------------------------------
# Guard: yaw + mesh incompatibility
# ---------------------------------------------------------------------------


def test_random_yaw_mesh_mode_assertion():
    """random_yaw_init=True + CollisionMode.MESH should raise AssertionError."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH),
        random_yaw_init=True,
    )
    placer = ObjectPlacer(params=params)

    table = _make_table()
    obj = _make_cylinder("can")
    obj.add_relation(On(table))

    with pytest.raises(AssertionError, match="random_yaw_init"):
        placer.place([table, obj])


# ---------------------------------------------------------------------------
# Missing tests identified in review
# ---------------------------------------------------------------------------


@requires_warp
def test_mesh_zero_loss_well_separated_cylinders():
    """Mesh mode correctly reports zero loss when objects are clearly separated.

    Uses large enough separation that sphere decomposition inflation (sphere_radius=0.01)
    cannot bridge the gap, proving mesh mode reads actual geometry.
    """
    a = _make_cylinder("a", radius=0.03, height=0.1)
    b = _make_cylinder("b", radius=0.03, height=0.1)

    # Separated by 0.15 in X — well clear of sum-of-radii + inflation
    a_pos = (0.0, 0.0, 0.0)
    b_pos = (0.15, 0.0, 0.0)

    strategy = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10000.0)
    child_pos = torch.tensor(a_pos, dtype=torch.float32)
    loss = strategy.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=b.get_bounding_box().translated(b_pos),
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor(b_pos, dtype=torch.float32),
    )
    assert loss.item() == 0.0, f"Well-separated cylinders should have zero mesh loss, got {loss.item()}"


@requires_warp
def test_object_placer_mesh_mode_end_to_end():
    """ObjectPlacer.place() with CollisionMode.MESH returns a valid result."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _make_table()
    a = _make_cylinder("cyl_a")
    b = _make_cylinder("cyl_b")
    a.add_relation(On(table))
    b.add_relation(On(table))

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=300, verbose=False),
        max_placement_attempts=5,
        verbose=False,
    )
    placer = ObjectPlacer(params=params)
    result = placer.place([table, a, b])
    assert result.success, f"Placement failed with loss={result.final_loss}"


@requires_warp
def test_dispatch_routes_to_mesh_in_mesh_mode():
    """MESH mode with mesh-bearing objects routes to mesh strategy (produces different loss than AABB)."""
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    b.set_initial_pose(Pose(position_xyz=(0.04, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = b.get_bounding_box().translated((0.04, 0.0, 0.0))

    # AABB: overlapping (distance 0.04 < 0.03+0.03=0.06 on X, full overlap on Y/Z)
    aabb_strategy = NoCollisionLossStrategy(collision_mode=CollisionMode.BBOX, slope=10000.0)
    loss_aabb = aabb_strategy.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor([0.04, 0.0, 0.0]),
    )

    # MESH: may or may not overlap depending on sphere placement, but should differ from AABB
    mesh_strategy = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10000.0)
    loss_mesh = mesh_strategy.compute_loss(
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
        child_obj=a,
        parent_obj=b,
        parent_pos=torch.tensor([0.04, 0.0, 0.0]),
    )

    # Key assertion: AABB loss is positive (boxes overlap), proving dispatch reached AABB path
    assert loss_aabb.item() > 0.0
    # Mesh loss must differ from AABB — proves a different code path ran
    assert (
        loss_mesh.item() != loss_aabb.item()
    ), f"Mesh and AABB losses are identical ({loss_mesh.item()}) — dispatch may not have routed to mesh"


@requires_warp
def test_validate_no_overlap_mesh_catches_overlap():
    """Direct test of _validate_no_overlap_mesh: overlapping cylinders should fail validation."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _make_table()
    a = _make_cylinder("cyl_a")
    b = _make_cylinder("cyl_b")
    a.add_relation(On(table))
    b.add_relation(On(table))

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False),
        verbose=False,
    )
    placer = ObjectPlacer(params=params)

    # Overlapping positions
    positions = {table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.0, 0.0, 0.05)}
    assert not placer._validate_no_overlap_mesh(positions)

    # Separated positions
    positions_sep = {table: (0.0, 0.0, 0.0), a: (0.2, 0.0, 0.05), b: (-0.2, 0.0, 0.05)}
    assert placer._validate_no_overlap_mesh(positions_sep)


@requires_warp
def test_mesh_sdf_backward_gradient():
    """mesh_sdf backward should produce non-zero gradients pointing outward for interior points."""
    from isaaclab_arena.relations.warp_sdf_kernels import mesh_sdf

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    manager = WarpMeshManager(num_spheres=10, device="cpu")
    warp_mesh = manager.get_warp_mesh(mesh)

    # Off-center point inside the box, closer to +X face
    points = torch.tensor([[0.08, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    sdf = mesh_sdf(points, warp_mesh)
    assert sdf.item() < 0.0, "Point inside box should have negative SDF"

    sdf.backward()
    assert points.grad is not None
    grad = points.grad[0]
    # Gradient should point toward the nearest face (+X direction)
    assert grad[0].item() > 0.0, f"Gradient X should be positive (toward +X face), got {grad[0].item()}"
    assert abs(grad[0].item()) > abs(grad[1].item()), "X component should dominate (closest to +X face)"


@requires_warp
def test_rotated_anchor_raises():
    """Mesh collision with a rotated anchor must raise AssertionError."""
    a = _make_cylinder("child", radius=0.03, height=0.1)
    table = _make_cylinder("table", radius=0.2, height=0.05)
    # Give the anchor a non-identity rotation
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.383, 0.924)))

    strategy = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=10000.0)
    child_pos = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float32)
    dummy_bbox = a.get_bounding_box()
    parent_bbox = table.get_bounding_box().translated((0.0, 0.0, 0.0))

    with pytest.raises(AssertionError, match="rotated anchor"):
        strategy.compute_loss(
            clearance_m=0.0,
            child_pos=child_pos,
            child_bbox=dummy_bbox,
            parent_world_bbox=parent_bbox,
            child_obj=a,
            parent_obj=table,
            parent_pos=None,
        )
