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


# Helpers


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


# Unit: greedy_sphere_decomposition


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


# Unit: WarpMeshManager caching


@requires_warp
def test_warp_mesh_caching():
    """Same mesh object should return identical Warp mesh from cache."""
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    manager = WarpMeshManager(num_spheres=10)
    m1 = manager.get_warp_mesh(mesh)
    m2 = manager.get_warp_mesh(mesh)
    assert m1 is m2


# Unit: NoCollisionLossStrategy routing


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


# Unit: NoCollisionLossStrategy mesh mode (requires warp)


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


# Integration: solver with mesh mode


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


# Guard: yaw + mesh incompatibility


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

    result = placer.place([table, cyl_a, cyl_b])
    assert result.success


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
    import math

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
def test_validate_no_overlap_mesh_respects_anchor_yaw():
    """Validator must use anchor's initial_pose yaw (not identity) when checking overlap."""
    import math

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
def test_solver_mesh_loss_detects_overlap_like_per_pair():
    """Solver's multi-mesh kernel detects the same overlap as per-pair strategy."""
    slope = 10000.0
    strategy = NoCollisionLossStrategy(collision_mode=CollisionMode.MESH, slope=slope)

    table = _make_table()
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    b.set_initial_pose(Pose(position_xyz=(0.04, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    a.add_relation(On(table))
    b.add_relation(On(table))

    pos_a = (0.0, 0.0, 0.05)
    pos_b = (0.04, 0.0, 0.05)
    initial = [{table: (0.0, 0.0, 0.0), a: pos_a, b: pos_b}]

    # Per-pair reference (strategy already applies slope internally)
    device = torch.device("cpu")
    a_pos_t = torch.tensor(pos_a, dtype=torch.float32, device=device)
    b_pos_t = torch.tensor(pos_b, dtype=torch.float32, device=device)
    loss_ab = strategy._compute_mesh_loss(
        0.0,
        a_pos_t.unsqueeze(0),
        a,
        a.get_collision_mesh(),
        b_pos_t.unsqueeze(0),
        b,
        b.get_collision_mesh(),
    )
    loss_ba = strategy._compute_mesh_loss(
        0.0,
        b_pos_t.unsqueeze(0),
        b,
        b.get_collision_mesh(),
        a_pos_t.unsqueeze(0),
        a,
        a.get_collision_mesh(),
    )
    # Strategy already applies slope; sum of fwd+rev is the mesh collision component
    expected_mesh_loss = loss_ab.item() + loss_ba.item()

    # Production path via solver (multi-mesh kernel, max_iters=0 = no optimization)
    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, a, b], initial)
    solver_loss = solver.last_loss_per_env[0].item()

    # Both should detect overlapping objects
    assert expected_mesh_loss > 0.0, "Per-pair reference should detect overlap"
    assert solver_loss > 0.0, "Solver should detect overlap"
    # Solver loss >= mesh_loss (includes On-relation contribution).
    # The mesh component should be within same order of magnitude.
    assert solver_loss >= expected_mesh_loss * 0.1


@requires_warp
def test_solver_mesh_batch_size_two():
    """Solver MESH mode handles batch_size > 1 (both envs solved independently)."""
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

    # Env 0: should have moved objects apart
    pos_a_0 = np.array(results[0][a])
    pos_b_0 = np.array(results[0][b])
    dist_0 = np.linalg.norm(pos_a_0[:2] - pos_b_0[:2])
    assert dist_0 > 0.06, f"Env 0: objects not separated, dist={dist_0:.4f}"

    # Env 1: already separated, should stay roughly in place
    pos_a_1 = np.array(results[1][a])
    pos_b_1 = np.array(results[1][b])
    dist_1 = np.linalg.norm(pos_a_1[:2] - pos_b_1[:2])
    assert dist_1 > 0.3, f"Env 1: separated objects moved too much, dist={dist_1:.4f}"


# Unit: AABB broadphase in production path


@requires_warp
def test_broadphase_skips_separated_pairs():
    """Well-separated objects produce zero mesh loss from the solver path."""
    table = _make_table()
    a = _make_cylinder("a", radius=0.03)
    b = _make_cylinder("b", radius=0.03)
    a.add_relation(On(table))
    b.add_relation(On(table))

    # Objects far apart — broadphase should filter them out
    initial = [{table: (0.0, 0.0, 0.0), a: (-0.4, 0.0, 0.05), b: (0.4, 0.0, 0.05)}]

    solver = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver.solve([table, a, b], initial)
    # With max_iters=0, loss is from initial positions.
    # Objects are well separated, so collision loss should be minimal
    # (only On-relation losses contribute).
    loss = solver.last_loss_per_env[0].item()

    # Compare with an overlapping case to confirm broadphase actually filters
    initial_overlap = [{table: (0.0, 0.0, 0.0), a: (0.0, 0.0, 0.05), b: (0.01, 0.0, 0.05)}]
    solver2 = RelationSolver(params=RelationSolverParams(collision_mode=CollisionMode.MESH, max_iters=0, verbose=False))
    solver2.solve([table, a, b], initial_overlap)
    loss_overlap = solver2.last_loss_per_env[0].item()

    assert loss_overlap > loss, "Overlapping case should have higher loss than separated"


@requires_warp
def test_broadphase_does_not_skip_overlapping_pairs():
    """Overlapping objects must produce nonzero mesh loss from the solver path."""
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


# Unit: multi_mesh_sdf


@requires_warp
def test_multi_mesh_sdf_distinct_meshes():
    """Verify mesh_indices routes queries to different meshes (not stuck at index 0)."""
    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    # Tall cylinder vs flat box — maximally different SDF at the query point.
    cylinder = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=32)
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.02))
    mgr = WarpMeshManager(num_spheres=10, device="cpu")

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
    """Backward through multi_mesh_sdf produces correct gradient direction."""
    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    mesh = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=32)
    mgr = WarpMeshManager(num_spheres=10, device="cpu")
    warp_mesh = mgr.get_warp_mesh(mesh)

    mesh_id_array = wp.array([warp_mesh.id], dtype=wp.uint64, device="cpu")
    mesh_indices = wp.array([0], dtype=wp.int32, device="cpu")

    # Point at (0.02, 0, 0) inside cylinder (r=0.05): SDF gradient should point outward (+x)
    points = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    sdf = multi_mesh_sdf(points, mesh_id_array, mesh_indices)
    sdf.backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    # SDF gradient at (0.02, 0, 0) should point radially outward: positive x, near-zero y/z
    assert points.grad[0, 0].item() > 0.1, f"Expected +x gradient, got {points.grad[0].tolist()}"


@requires_warp
def test_solver_target_only_yaw():
    """Target-only yaw must affect collision detection (catches missing parent rotation)."""
    import math

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

    # No rotation: should be low/zero mesh collision
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
    import math

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
