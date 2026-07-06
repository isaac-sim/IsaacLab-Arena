# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh-based collision detection: sphere decomposition, dispatch, and end-to-end solver."""

from __future__ import annotations

import math
import numpy as np
import torch
import trimesh

import pytest

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache, greedy_sphere_decomposition
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

try:
    import warp as wp

    wp.init()
    _WARP_AVAILABLE = True
except Exception:
    _WARP_AVAILABLE = False

requires_warp = pytest.mark.skipif(not _WARP_AVAILABLE, reason="Warp not available")


# Unit tests


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


def test_sphere_decomposition_covers_surface():
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=32)
    spheres = greedy_sphere_decomposition(mesh, num_spheres=20, n_surface=500)
    assert spheres.shape[1] == 4
    assert spheres.shape[0] <= 20

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


@requires_warp
def test_warp_mesh_caching():
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    manager = WarpMeshAndSphereCache(num_spheres=10)
    m1 = manager.get_warp_mesh(mesh)
    m2 = manager.get_warp_mesh(mesh)
    assert m1 is m2


def _batched_aabb_loss(strategy, clearance_m, child_pos, child_bbox, parent_world_bbox):
    """Helper: compute single-pair AABB loss via compute_loss_batched."""
    subject_min = (child_pos + child_bbox.min_point).unsqueeze(0).unsqueeze(0)
    subject_max = (child_pos + child_bbox.max_point).unsqueeze(0).unsqueeze(0)
    obstacle_min = parent_world_bbox.min_point.unsqueeze(0)
    obstacle_max = parent_world_bbox.max_point.unsqueeze(0)
    loss = strategy.compute_loss_batched(clearance_m, subject_min, subject_max, obstacle_min, obstacle_max)
    return loss.squeeze()


def test_aabb_zero_loss_well_separated():
    strategy = NoCollisionLossStrategy(slope=10.0)

    obj_a = _make_cylinder("a")
    obj_b = _make_cylinder("b")

    loss = _batched_aabb_loss(
        strategy,
        clearance_m=0.0,
        child_pos=torch.tensor([0.0, 0.0, 0.0]),
        child_bbox=obj_a.get_bounding_box(),
        parent_world_bbox=obj_b.get_bounding_box().translated((0.5, 0.0, 0.0)),
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_aabb_positive_loss_fully_overlapping():
    strategy = NoCollisionLossStrategy(slope=10000.0)
    a = _make_cylinder("a")
    b = _make_cylinder("b")

    loss = _batched_aabb_loss(
        strategy,
        clearance_m=0.0,
        child_pos=torch.tensor([0.0, 0.0, 0.0]),
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=b.get_bounding_box().translated((0.0, 0.0, 0.0)),
    )
    assert loss.item() > 0.0


def test_aabb_clearance_m_increases_loss():
    strategy = NoCollisionLossStrategy(slope=10000.0)
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    child_pos = torch.tensor([0.0, 0.0, 0.0])
    parent_world_bbox = b.get_bounding_box().translated((0.07, 0.0, 0.0))

    loss_no_clearance = _batched_aabb_loss(
        strategy,
        clearance_m=0.0,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
    )
    loss_with_clearance = _batched_aabb_loss(
        strategy,
        clearance_m=0.05,
        child_pos=child_pos,
        child_bbox=a.get_bounding_box(),
        parent_world_bbox=parent_world_bbox,
    )
    assert loss_with_clearance.item() > loss_no_clearance.item()


# Integration tests


@requires_warp
def test_solver_separates_overlapping_cylinders_mesh_mode():
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

    z = result[obj][2]
    assert 0.0 < z < 0.15, f"Object pushed too far: z={z}"


# Guard tests


@requires_warp
def test_random_yaw_mesh_mode_places_successfully():
    """random_yaw_init=True + CollisionMode.MESH should place objects without error."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(
            collision_mode=CollisionMode.MESH,
            max_iters=50,
        ),
        random_yaw_init=True,
    )
    placer = ObjectPlacer(params=params)

    table = _make_table()
    cyl_a = _make_cylinder("a")
    cyl_b = _make_cylinder("b")
    cyl_a.add_relation(On(table))
    cyl_b.add_relation(On(table))

    results = placer.place([table, cyl_a, cyl_b])
    assert results[0].success


@requires_warp
def test_anchor_with_rotate_around_solution_rejected():
    """Anchor + RotateAroundSolution must fail loudly (not silently mismatch)."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import RotateAroundSolution

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=5),
        random_yaw_init=True,
    )
    placer = ObjectPlacer(params=params)

    table = _make_table()
    table.add_relation(RotateAroundSolution(yaw_rad=0.5))
    child = _make_cylinder("child")
    child.add_relation(On(table))

    with pytest.raises(AssertionError, match="Anchor.*RotateAroundSolution"):
        placer.place([table, child])


@requires_warp
def test_centers_in_target_frame_applies_both_yaws():
    """Net yaw = source - target; equal yaws cancel out."""

    from isaaclab_arena.relations.object_placer import ObjectPlacer

    src = DummyObject(
        "src",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        collision_mesh=trimesh.creation.box(extents=(0.2, 0.2, 0.2)),
    )
    tgt = DummyObject(
        "tgt",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1)),
        collision_mesh=trimesh.creation.box(extents=(0.2, 0.2, 0.2)),
    )
    centers = torch.tensor([[0.10, 0.0, 0.0]])
    src_pos = torch.tensor([0.0, 0.0, 0.0])
    tgt_pos = torch.tensor([0.0, 0.0, 0.0])

    # No orientations: pass-through
    result = ObjectPlacer._centers_in_target_frame(centers, src, tgt, src_pos, tgt_pos, None)
    assert torch.allclose(result, centers, atol=1e-6)

    # Source yaw=pi/2, target yaw=0: net rotation = pi/2
    result = ObjectPlacer._centers_in_target_frame(centers, src, tgt, src_pos, tgt_pos, {src: math.pi / 2})
    assert abs(result[0, 0].item()) < 1e-5
    assert abs(result[0, 1].item() - 0.10) < 1e-5

    # Both at same yaw: net rotation = 0, centers unchanged (offset is zero here)
    result = ObjectPlacer._centers_in_target_frame(
        centers, src, tgt, src_pos, tgt_pos, {src: math.pi / 2, tgt: math.pi / 2}
    )
    assert torch.allclose(result, centers, atol=1e-5)


@requires_warp
def test_object_placer_mesh_mode_end_to_end():
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
    results = placer.place([table, a, b])
    assert results[0].success, f"Placement failed with loss={results[0].final_loss}"


@requires_warp
def test_validate_no_overlap_mesh_catches_overlap():
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
def test_validate_no_overlap_mesh_sentinel_fails(monkeypatch):
    """A sentinel SDF (no resolvable face) must fail validation, not certify collision-free."""
    from isaaclab_arena.relations import warp_sdf_kernels
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
    positions = {table: (0.0, 0.0, 0.0), a: (0.2, 0.0, 0.05), b: (-0.2, 0.0, 0.05)}
    assert placer._validate_no_overlap_mesh(positions)

    # Force every query to hit the sentinel; the same separated layout must now fail.
    from isaaclab_arena.relations import object_placer as _op_mod

    real_mesh_sdf = warp_sdf_kernels.mesh_sdf

    def fake_sdf(points, mesh):
        return torch.full_like(real_mesh_sdf(points, mesh), 1.0e6)

    monkeypatch.setattr(warp_sdf_kernels, "mesh_sdf", fake_sdf)
    monkeypatch.setattr(_op_mod, "mesh_sdf", fake_sdf)
    assert not placer._validate_no_overlap_mesh(positions)


@requires_warp
def test_validate_no_overlap_mesh_respects_anchor_yaw():
    """Validator must use anchor's initial_pose yaw (not identity) when checking overlap."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _make_table()
    # Long thin anchor rotated 90° about Z
    anchor_mesh = trimesh.creation.box(extents=(0.2, 0.02, 0.05))
    anchor = DummyObject(
        "anchor",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.01, -0.025), max_point=(0.1, 0.01, 0.025)),
        collision_mesh=anchor_mesh,
    )
    sz = math.sin(math.pi / 4)
    cz = math.cos(math.pi / 4)
    anchor.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, sz, cz)))
    anchor.add_relation(IsAnchor())

    child = _make_cylinder("child", radius=0.012)
    child.add_relation(On(table))

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False),
        verbose=False,
    )
    placer = ObjectPlacer(params=params)

    # Child at Y=0.02: outside unrotated anchor (half-width=0.01),
    # but inside rotated anchor (half-length=0.1 now spans Y).
    positions = {table: (0.0, 0.0, 0.0), anchor: (0.0, 0.0, 0.05), child: (0.0, 0.02, 0.05)}
    assert not placer._validate_no_overlap_mesh(positions), "Validator should detect overlap with yawed anchor"


@requires_warp
def test_mesh_sdf_backward_gradient():
    """mesh_sdf backward gradient points toward the nearest face for an interior point."""
    from isaaclab_arena.relations.warp_sdf_kernels import mesh_sdf

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    manager = WarpMeshAndSphereCache(num_spheres=10, device="cpu")
    warp_mesh = manager.get_warp_mesh(mesh)

    # Off-center point inside the box, closer to +X face
    points = torch.tensor([[0.08, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    sdf = mesh_sdf(points, warp_mesh)
    assert sdf.item() < 0.0, "Point inside box should have negative SDF"

    sdf.backward()
    assert points.grad is not None
    grad = points.grad[0]
    assert grad[0].item() > 0.0, f"Gradient X should be positive (toward +X face), got {grad[0].item()}"
    assert abs(grad[0].item()) > abs(grad[1].item()), "X component should dominate (closest to +X face)"


@requires_warp
def test_solver_mesh_batch_size_two():
    table = _make_table()
    a = _make_cylinder("cyl_a")
    b = _make_cylinder("cyl_b")
    a.add_relation(On(table))
    b.add_relation(On(table))

    # Env 0: overlapping, Env 1: separated
    initial = [
        {table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.01, 0.0, 0.05)},
        {table: (0.0, 0.0, 0.0), a: (-0.2, 0.0, 0.05), b: (0.2, 0.0, 0.05)},
    ]

    solver = RelationSolver(
        params=RelationSolverParams(
            collision_mode=CollisionMode.MESH, max_iters=200, convergence_threshold=1e-4, verbose=False
        )
    )
    results = solver.solve([table, a, b], initial)
    assert len(results) == 2

    pos_a_0 = np.array(results[0][a])
    pos_b_0 = np.array(results[0][b])
    dist_0 = np.linalg.norm(pos_a_0[:2] - pos_b_0[:2])
    assert dist_0 > 0.06, f"Env 0: objects not separated, dist={dist_0:.4f}"

    pos_a_1 = np.array(results[1][a])
    pos_b_1 = np.array(results[1][b])
    dist_1 = np.linalg.norm(pos_a_1[:2] - pos_b_1[:2])
    assert dist_1 > 0.3, f"Env 1: separated objects moved too much, dist={dist_1:.4f}"


@requires_warp
def test_broadphase_skips_separated_pairs():
    table = _make_table()
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    a.add_relation(On(table))
    b.add_relation(On(table))

    # Objects far apart — broadphase should filter them out
    initial = [{table: (0.0, 0.0, 0.0), a: (-0.4, 0.0, 0.05), b: (0.4, 0.0, 0.05)}]

    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, a, b], initial)
    loss = solver.last_loss_per_env[0].item()

    # Compare with an overlapping case to confirm broadphase actually filters
    initial_overlap = [{table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.01, 0.0, 0.05)}]
    solver2 = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver2.solve([table, a, b], initial_overlap)
    loss_overlap = solver2.last_loss_per_env[0].item()

    assert loss_overlap > loss, "Overlapping case should have higher loss than separated"


@requires_warp
def test_broadphase_does_not_skip_overlapping_pairs():
    table = _make_table()
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    a.add_relation(On(table))
    b.add_relation(On(table))

    # Overlapping: at same position
    initial = [{table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.0, 0.0, 0.05)}]

    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, a, b], initial)
    loss = solver.last_loss_per_env[0].item()
    assert loss > 0.0, "Overlapping objects should produce nonzero loss"


@requires_warp
def test_multi_mesh_sdf_distinct_meshes():
    """Regression: mesh_indices routing could silently query only mesh 0."""
    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    # Tall cylinder vs flat box — maximally different SDF at the query point.
    cylinder = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=32)
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.02))
    mgr = WarpMeshAndSphereCache(num_spheres=10, device="cpu")

    warp_cyl = mgr.get_warp_mesh(cylinder)
    warp_box = mgr.get_warp_mesh(box)

    mesh_id_array = wp.array([warp_cyl.id, warp_box.id], dtype=wp.uint64, device="cpu")

    # Point at origin: inside cylinder (depth ~0.05), inside box (depth ~0.01)
    p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    idx_cyl = wp.array([0], dtype=wp.int32, device="cpu")
    sdf_cyl = multi_mesh_sdf(p, mesh_id_array, idx_cyl)

    idx_box = wp.array([1], dtype=wp.int32, device="cpu")
    sdf_box = multi_mesh_sdf(p, mesh_id_array, idx_box)

    # Both inside (negative), but cylinder is much deeper
    assert sdf_cyl.item() < -0.03, f"Expected deep inside cylinder, got {sdf_cyl.item()}"
    assert sdf_box.item() < 0.0
    assert sdf_cyl.item() < sdf_box.item(), "Cylinder should be deeper than flat box at origin"


@requires_warp
def test_multi_mesh_sdf_backward():
    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    mesh = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=32)
    mgr = WarpMeshAndSphereCache(num_spheres=10, device="cpu")
    warp_mesh = mgr.get_warp_mesh(mesh)

    mesh_id_array = wp.array([warp_mesh.id], dtype=wp.uint64, device="cpu")
    mesh_indices = wp.array([0], dtype=wp.int32, device="cpu")

    # Point at (0.02, 0, 0): inside cylinder (r=0.05), closest face in +X direction.
    points = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    sdf = multi_mesh_sdf(points, mesh_id_array, mesh_indices)
    sdf.backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert points.grad[0, 0].item() > 0.1, f"Expected +x gradient, got {points.grad[0].tolist()}"


@requires_warp
def test_solver_target_only_yaw():
    """Target-only yaw must affect collision detection (catches missing parent rotation)."""
    table = _make_table()
    # Long thin box: if rotated 90° around Z, its collision footprint changes axis
    target = _make_box_obj("target", sx=0.2, sy=0.02, sz=0.05)
    target.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.05)))
    target.add_relation(IsAnchor())

    child = _make_cylinder("child", radius=0.015)
    child.add_relation(On(table))

    # Place child next to target's long axis (Y=0.03). Without target rotation,
    # this is well outside the 0.02/2 half-width in Y → no collision.
    # With target rotated 90°, the 0.2/2=0.1 half-extent now spans Y → collision.
    initial = [{table: (0.0, 0.0, 0.0), target: (0.0, 0.0, 0.05), child: (0.0, 0.03, 0.05)}]

    solver_no_rot = RelationSolver(
        params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False)
    )
    solver_no_rot.solve([table, target, child], initial, orientations=None)
    loss_no_rot = solver_no_rot.last_loss_per_env[0].item()

    # Target rotated 90° around Z: child is now inside target's mesh
    orientations_rotated = [{target: math.pi / 2, child: 0.0}]
    solver_rot = RelationSolver(
        params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False)
    )
    solver_rot.solve([table, target, child], initial, orientations=orientations_rotated)
    loss_rot = solver_rot.last_loss_per_env[0].item()

    assert (
        loss_rot > loss_no_rot + 1.0
    ), f"Target yaw=90° should dramatically increase collision loss (got {loss_rot:.2f} vs {loss_no_rot:.2f})"


@requires_warp
def test_anchor_initial_pose_yaw_affects_collision():
    """Anchor Z-yaw baked in initial_pose (no orientations dict) must affect SDF queries."""
    table = _make_table()
    # Long thin anchor: 0.2 x 0.02 — rotation changes which axis is long
    target_mesh = trimesh.creation.box(extents=(0.2, 0.02, 0.05))
    target = DummyObject(
        "target",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.01, -0.025), max_point=(0.1, 0.01, 0.025)),
        collision_mesh=target_mesh,
    )
    # Bake 90° Z-yaw into initial_pose (not via orientations dict)
    sz = math.sin(math.pi / 4)
    cz = math.cos(math.pi / 4)
    target.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, sz, cz)))
    target.add_relation(IsAnchor())

    child = _make_cylinder("child", radius=0.012)
    child.add_relation(On(table))

    # Child at Y=0.02: outside unrotated target (half-width=0.01), inside rotated (half-length=0.1)
    initial = [{table: (0.0, 0.0, 0.0), target: (0.0, 0.0, 0.05), child: (0.0, 0.02, 0.05)}]

    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, target, child], initial, orientations=None)
    loss_yawed = solver.last_loss_per_env[0].item()

    # Same geometry with identity anchor — should have lower loss
    target_id = DummyObject(
        "target_id",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.1, -0.01, -0.025), max_point=(0.1, 0.01, 0.025)),
        collision_mesh=target_mesh,
    )
    target_id.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.05)))
    target_id.add_relation(IsAnchor())
    initial_id = [{table: (0.0, 0.0, 0.0), target_id: (0.0, 0.0, 0.05), child: (0.0, 0.02, 0.05)}]

    solver_id = RelationSolver(
        params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False)
    )
    solver_id.solve([table, target_id, child], initial_id, orientations=None)
    loss_identity = solver_id.last_loss_per_env[0].item()

    assert loss_yawed > loss_identity + 1.0, (
        "Yawed anchor (from initial_pose) should produce higher collision "
        f"(got {loss_yawed:.2f} vs identity {loss_identity:.2f})"
    )


@requires_warp
def test_aabb_gate_does_not_reject_diagonal_cylinders():
    """Regression: MESH-mode validator accepts cylinders whose AABBs overlap but meshes don't."""
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    table = _make_table()
    # r=0.05, b at (0.09, 0.09): AABB overlap (0.09 < 2*0.05=0.10) but geometric
    # distance = 0.127 > sum-of-radii 0.10 + sphere_radius 0.01, plenty of margin.
    a = _make_cylinder("a", radius=0.05, height=0.1)
    b = _make_cylinder("b", radius=0.05, height=0.1)
    a.add_relation(On(table))
    b.add_relation(On(table))

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False),
        verbose=False,
    )
    placer = ObjectPlacer(params=params)

    positions = {table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.09, 0.09, 0.05)}

    # Sanity: AABB check without skip_mesh_pairs REJECTS this layout
    env_bboxes = {obj: obj.get_bounding_box() for obj in positions}
    assert not placer._validate_no_overlap(
        positions, env_bboxes, skip_mesh_pairs=False
    ), "Sanity check failed: AABB should reject diagonal cylinders"

    # With skip_mesh_pairs=True (MESH mode), AABB validator skips this pair
    assert placer._validate_no_overlap(
        positions, env_bboxes, skip_mesh_pairs=True
    ), "AABB validator with skip_mesh_pairs should accept this pair"

    # Mesh validator accepts (cylinders don't actually overlap)
    assert placer._validate_no_overlap_mesh(
        positions
    ), "Mesh validator should accept diagonal cylinders that don't geometrically overlap"


def test_broadphase_does_not_falsely_cull_yawed_elongated_pair():
    """Elongated objects at ~90° yaw whose unrotated AABBs are separated must still produce loss.

    Regression for double-rotation in the broadphase: if bboxes were rotated twice,
    the second rotation could recover the unrotated shape and falsely mark the pair separated.
    """
    table = _make_table()
    # Elongated boxes: 0.4m long, 0.02m wide
    a_mesh = trimesh.creation.box(extents=(0.4, 0.02, 0.05))
    b_mesh = trimesh.creation.box(extents=(0.4, 0.02, 0.05))
    a = DummyObject(
        "a",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.2, -0.01, -0.025), max_point=(0.2, 0.01, 0.025)),
        collision_mesh=a_mesh,
    )
    b = DummyObject(
        "b",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.2, -0.01, -0.025), max_point=(0.2, 0.01, 0.025)),
        collision_mesh=b_mesh,
    )
    a.add_relation(On(table))
    b.add_relation(On(table))

    # Place b offset in X by 0.05: unrotated AABBs overlap (half-width 0.2+0.2 > 0.05),
    # but if we yaw both by pi/2 the unrotated AABB (width 0.02) would be separated.
    yaw = math.pi / 2
    initial = [{table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.05, 0.0, 0.05)}]
    orientations = [{a: yaw, b: yaw}]

    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, a, b], initial, orientations=orientations)
    loss = solver.last_loss_per_env[0].item()
    assert loss > 0.0, "Broadphase must not falsely cull yawed elongated pairs that genuinely collide"
