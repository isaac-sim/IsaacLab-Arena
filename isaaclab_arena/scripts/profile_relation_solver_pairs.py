# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Profile relation-solver loss terms and mesh-pair costs for a graph-spec YAML."""

from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device, repeats: int = 1) -> float:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3 / repeats


def _run_profile(
    yaml_path: str,
    placement_seed: int,
    candidates: int,
    profile_iters: int,
    mesh_impl: str = "batched",
) -> dict[str, Any]:
    import torch

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.relation_solver_interface import (
        _get_passive_collision_objects,
        _should_include_background_mesh,
    )
    from isaaclab_arena.relations import no_overlap_mesh as no_overlap_mesh_mod
    from isaaclab_arena.relations import relation_solver as relation_solver_mod
    from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
    from isaaclab_arena.relations.no_overlap_aabb import compute_no_overlap_loss_aabb
    from isaaclab_arena.relations.no_overlap_mesh import (
        _compute_no_overlap_loss_mesh_serial,
        compute_no_overlap_loss_mesh,
    )
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relation_solver_state import RelationSolverState
    from isaaclab_arena.relations.relations import Relation, UnaryRelation, get_anchor_objects
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    assert mesh_impl in {"batched", "serial"}, f"Unknown mesh_impl={mesh_impl}"
    mesh_loss_fn = compute_no_overlap_loss_mesh if mesh_impl == "batched" else _compute_no_overlap_loss_mesh_serial
    # Keep RelationSolver on the same implementation as the timed mesh term.
    relation_solver_mod.compute_no_overlap_loss_mesh = mesh_loss_fn
    no_overlap_mesh_mod.compute_no_overlap_loss_mesh = mesh_loss_fn

    spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
    arena_env = build_arena_env_from_graph_spec(spec)
    placer_params = arena_env.placer_params or ObjectPlacerParams()
    placer_params = copy.copy(placer_params)
    placer_params.placement_seed = placement_seed
    placer_params.solver_params = RelationSolverParams(
        verbose=False,
        save_position_history=False,
        profile=True,
        max_iters=profile_iters,
        collision_mode=placer_params.solver_params.collision_mode,
        num_spheres=placer_params.solver_params.num_spheres,
        clearance_m=placer_params.solver_params.clearance_m,
    )

    placement_assets = []
    for asset in arena_env.scene.assets.values():
        if asset.get_relations():
            placement_assets.append(asset)
        if isinstance(asset, ObjectReference) and asset.parent_asset.get_relations():
            continue
    if arena_env.embodiment is not None and arena_env.embodiment.get_relations():
        placement_assets.append(arena_env.embodiment)

    scene_assets = list(arena_env.scene.assets.values())
    collision_objects = _get_passive_collision_objects(
        scene_assets,
        include_background=_should_include_background_mesh(
            placement_assets, scene_assets, placer_params.solver_params.collision_mode
        ),
    )

    placer = ObjectPlacer(params=placer_params)
    anchors = set(get_anchor_objects(placement_assets))
    num_envs = 1
    assign_variants_for_envs(placement_assets, num_envs, placement_seed=placement_seed)
    env_bboxes = build_per_env_bounding_boxes(placement_assets, num_envs)
    unrotated = env_bboxes.get_bounding_boxes_for_solver_candidates(candidates)
    per_env = env_bboxes.get_bounding_boxes_for_all_envs()

    generator = torch.Generator()
    initial_positions = []
    orientations = []
    for i in range(candidates):
        generator.manual_seed(placement_seed + i)
        initial_positions.append(placer._generate_initial_positions(placement_assets, anchors, per_env[0], generator))
        orientations.append(placer._generate_initial_orientations(placement_assets, anchors, generator))
    candidate_bboxes = placer._rotate_candidate_bboxes(placement_assets, unrotated, orientations)

    # Mesh stats for collision objects / placement assets
    mesh_stats = []
    mgr_probe = WarpMeshAndSphereCache(num_spheres=placer_params.solver_params.num_spheres, device="cpu")
    for obj in [*placement_assets, *collision_objects]:
        mesh = mgr_probe.get_collision_mesh(obj) if hasattr(mgr_probe, "get_collision_mesh") else None
        if mesh is None and hasattr(obj, "get_collision_mesh"):
            try:
                mesh = obj.get_collision_mesh()
            except Exception:
                mesh = None
        if mesh is None:
            continue
        mesh_stats.append({
            "name": obj.name,
            "verts": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "extents": [float(x) for x in mesh.extents],
        })

    kitchen_mesh_breakdown = _kitchen_mesh_breakdown(arena_env)

    solver = RelationSolver(params=placer_params.solver_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warm / timed full solve
    solve_ms = _timed(
        lambda: solver.solve(
            placement_assets,
            initial_positions,
            env_bboxes=candidate_bboxes,
            env_bboxes_include_yaw=any(orientations),
            orientations=orientations,
            collision_objects=collision_objects,
        ),
        device,
    )
    # Re-solve once more to leave solver caches warm, then profile loss breakdown on current state
    positions = solver.solve(
        placement_assets,
        initial_positions,
        env_bboxes=candidate_bboxes,
        env_bboxes_include_yaw=any(orientations),
        orientations=orientations,
        collision_objects=collision_objects,
    )

    state = RelationSolverState(
        placement_assets,
        positions,
        device=device,
        env_bboxes=candidate_bboxes,
        collision_objects=collision_objects,
    )
    # Ensure mesh cache is ready (solve already did this, but rebuild from solver fields)
    assert solver._mesh_cache is not None or not solver._mesh_collision_enabled
    mesh_cache = solver._mesh_cache
    mesh_manager = solver._mesh_manager
    solver._mesh_orientations = orientations

    # ---- Per relation-term timing ----
    relation_rows = []
    for obj in state.optimizable_objects:
        for relation in obj.get_spatial_relations():
            child_pos = state.get_position(obj)
            child_bbox = state.get_bbox(obj)
            strategy = solver._get_strategy(relation)

            def _rel_fn(
                obj=obj,
                relation=relation,
                child_pos=child_pos,
                child_bbox=child_bbox,
                strategy=strategy,
            ):
                if isinstance(relation, UnaryRelation):
                    casted = strategy  # UnaryRelationLossStrategy
                    return casted.compute_loss(relation=relation, child_pos=child_pos, child_bbox=child_bbox)
                assert isinstance(relation, Relation)
                parent = relation.parent
                if parent in state.anchor_objects:
                    parent_world_bbox = state.get_fixed_obstacle_world_bbox(parent)
                else:
                    parent_pos = state.get_position(parent)
                    parent_bbox = state.get_bbox(parent)
                    parent_world_bbox = parent_bbox.translated(parent_pos)
                return strategy.compute_loss(
                    relation=relation,
                    child_pos=child_pos,
                    child_bbox=child_bbox,
                    parent_world_bbox=parent_world_bbox,
                )

            # Warm
            _rel_fn()
            ms = _timed(_rel_fn, device, repeats=20)
            loss_val = float(_rel_fn().mean().detach().cpu())
            parent_name = getattr(getattr(relation, "parent", None), "name", None)
            relation_rows.append({
                "kind": type(relation).__name__,
                "subject": obj.name,
                "reference": parent_name,
                "ms_per_call": ms,
                "loss_mean": loss_val,
                "category": "relation",
            })

    # ---- AABB no-overlap ----
    aabb_ms = _timed(
        lambda: compute_no_overlap_loss_aabb(
            state,
            solver._no_collision_strategy,
            placer_params.solver_params.clearance_m,
            mesh_manager,
            placer_params.solver_params.collision_mode,
            skip_mesh_pairs=solver._mesh_collision_enabled,
            debug=False,
        ),
        device,
        repeats=10,
    )
    relation_rows.append({
        "kind": "NoOverlapAABB",
        "subject": "*",
        "reference": "*",
        "ms_per_call": aabb_ms,
        "loss_mean": None,
        "category": "aabb_no_overlap",
    })

    # ---- Build-time placer validity (same checks as ObjectPlacer ranking) ----
    bboxes_per_candidate = [
        placer._get_bounding_boxes_for_candidate_index(candidate_bboxes, candidate_idx)
        for candidate_idx in range(candidates)
    ]
    validation_results = placer._validate_candidates(positions, orientations, bboxes_per_candidate, collision_objects)
    per_check_pass = {}
    for check in sorted({c for vr in validation_results for c in vr.validation_results}):
        per_check_pass[check] = sum(1 for vr in validation_results if vr.validation_results.get(check, False))
    valid_candidates = sum(1 for vr in validation_results if vr.do_all_required_validation_checks_pass())

    # ---- Full mesh no-overlap ----
    mesh_full_ms = None
    if mesh_cache is not None and mesh_manager is not None:
        mesh_full_ms = _timed(
            lambda: mesh_loss_fn(
                state,
                mesh_cache,
                mesh_manager,
                orientations,
                placer_params.solver_params.clearance_m,
                solver._no_collision_strategy.slope,
                False,
            ),
            device,
            repeats=5,
        )
        relation_rows.append({
            "kind": "NoOverlapMESH_all",
            "subject": "*",
            "reference": "*",
            "ms_per_call": mesh_full_ms,
            "loss_mean": None,
            "category": "mesh_no_overlap",
        })

        # Per-pair SDF cost: isolate spheres for one pair across all batch envs
        # Rebuild one-entry caches by slicing would be heavy; instead time SDF query
        # for each pair's spheres against its mesh, for all candidates.
        import warp as wp

        pair_rows = []
        for p in range(mesh_cache.num_pairs):
            subject = mesh_cache.pair_subject_objs[p]
            obstacle = mesh_cache.pair_obstacle_objs[p]
            sphere_mask = mesh_cache.sphere_pair_id == p
            n_spheres = int(sphere_mask.sum().item())
            centers_local = mesh_cache.all_centers_local[sphere_mask]  # (S, 3)
            mesh_idx = mesh_cache.sphere_mesh_idx[sphere_mask]
            mesh_ids = mesh_cache.mesh_id_array

            # Build query centers for all batch envs: local + (subject - obstacle)
            subj_pos = state.get_position(subject)  # (B, 3)
            if mesh_cache.pair_obstacle_is_fixed[p]:
                obs_pos = mesh_cache.pair_fixed_obstacle_pos[p].unsqueeze(0).expand(state.batch_size, 3)
            else:
                obs_pos = state.get_position(obstacle).detach()
            offsets = subj_pos - obs_pos  # (B, 3)
            # (B, S, 3)
            query = centers_local.unsqueeze(0) + offsets.unsqueeze(1)
            query_flat = query.reshape(-1, 3).contiguous()
            mesh_idx_flat = mesh_idx.repeat(state.batch_size)

            def _pair_sdf(
                query_flat=query_flat,
                mesh_ids=mesh_ids,
                mesh_idx_flat=mesh_idx_flat,
            ):
                idx_wp = wp.from_torch(mesh_idx_flat.contiguous(), dtype=wp.int32)
                return multi_mesh_sdf(query_flat, mesh_ids, idx_wp)

            _pair_sdf()
            sdf_ms = _timed(_pair_sdf, device, repeats=10)

            pair_rows.append({
                "kind": "MeshSDF",
                "subject": subject.name,
                "reference": obstacle.name,
                "obstacle_fixed": bool(mesh_cache.pair_obstacle_is_fixed[p]),
                "spheres_per_env": n_spheres,
                "queries_total": n_spheres * state.batch_size,
                "ms_per_call": sdf_ms,
                "us_per_query": (sdf_ms * 1e3) / max(1, n_spheres * state.batch_size),
                "category": "mesh_pair_sdf",
            })

        # Microbenchmark: Python overhead of compute_no_overlap_loss_mesh structure
        # (AABB filter + tensor construction) vs SDF kernel
        def _mesh_loss_once():
            return compute_no_overlap_loss_mesh(
                state,
                mesh_cache,
                mesh_manager,
                orientations,
                placer_params.solver_params.clearance_m,
                solver._no_collision_strategy.slope,
                False,
            )

        # Instrument sync hotspots inside one call by wrapping known sync ops
        sync_probe = _profile_sync_hotspots(state, mesh_cache, orientations, device)
        loop_probe = _profile_mesh_loop_phases(
            state,
            mesh_cache,
            mesh_manager,
            orientations,
            placer_params.solver_params.clearance_m,
            solver._no_collision_strategy.slope,
            device,
        )
    else:
        pair_rows = []
        sync_probe = {}
        loop_probe = {}

    # One-iteration total loss timing (matches solver step)
    def _total_loss():
        return solver._compute_total_loss(state)

    _total_loss()
    total_loss_ms = _timed(_total_loss, device, repeats=5)
    # with backward
    opt_pos = state.optimizable_positions

    def _fwd_bwd():
        if opt_pos.grad is not None:
            opt_pos.grad = None
        loss = solver._compute_total_loss(state)
        if loss.grad_fn is not None:
            loss.backward()
        return loss

    _fwd_bwd()
    fwd_bwd_ms = _timed(_fwd_bwd, device, repeats=5)

    all_rows = relation_rows + pair_rows
    all_rows.sort(key=lambda r: r["ms_per_call"], reverse=True)

    return {
        "yaml": yaml_path,
        "candidates": candidates,
        "profile_iters": profile_iters,
        "mesh_impl": mesh_impl,
        "device": str(device),
        "solve_wall_ms": solve_ms,
        "ms_per_iter_estimate": solve_ms / max(1, profile_iters),
        "total_loss_fwd_ms": total_loss_ms,
        "total_loss_fwd_bwd_ms": fwd_bwd_ms,
        "valid_candidates": valid_candidates,
        "total_candidates": candidates,
        "per_check_pass": per_check_pass,
        "mesh_collision_enabled": solver._mesh_collision_enabled,
        "num_mesh_pairs": 0 if mesh_cache is None else mesh_cache.num_pairs,
        "total_spheres": 0 if mesh_cache is None else mesh_cache.total_spheres,
        "mesh_stats": mesh_stats,
        "kitchen_mesh_breakdown": kitchen_mesh_breakdown,
        "sync_hotspots": sync_probe,
        "mesh_loop_phases_ms": loop_probe,
        "ranked_costs": all_rows,
    }


def _kitchen_mesh_breakdown(arena_env) -> dict[str, Any]:
    """Compare all Mesh prims vs CollisionAPI Mesh prims in the kitchen USD."""
    from pxr import Usd, UsdGeom, UsdPhysics

    kitchen = arena_env.scene.assets.get("kitchen")
    if kitchen is None or getattr(kitchen, "usd_path", None) is None:
        return {}
    stage = Usd.Stage.Open(kitchen.usd_path)
    all_v = all_f = all_n = 0
    col_v = col_f = col_n = 0
    gprim_col = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
            gprim_col += 1
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        if pts is None or fvc is None:
            continue
        n_tris = sum(max(0, int(c) - 2) for c in fvc)
        all_n += 1
        all_v += len(pts)
        all_f += n_tris
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_n += 1
            col_v += len(pts)
            col_f += n_tris
    return {
        "usd_path": kitchen.usd_path,
        "all_mesh_count": all_n,
        "all_mesh_verts": all_v,
        "all_mesh_tris_approx": all_f,
        "collision_mesh_count": col_n,
        "collision_mesh_verts": col_v,
        "collision_mesh_tris_approx": col_f,
        "collision_gprim_non_mesh": gprim_col,
        "visual_only_vert_fraction": (all_v - col_v) / max(1, all_v),
    }


def _profile_mesh_loop_phases(state, mesh_cache, mesh_manager, orientations, clearance_m, slope, device):
    """Break down compute_no_overlap_loss_mesh into phases over the full batch."""
    import torch

    import warp as wp

    from isaaclab_arena.relations.no_overlap_mesh import _rotate_bbox_extents
    from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, multi_mesh_sdf
    from isaaclab_arena.utils.yaw import rotate_points_by_yaw_batch

    num_pairs = mesh_cache.num_pairs
    B = state.batch_size
    phases = defaultdict(float)
    active_env_count = 0
    active_sphere_counts = []

    def run_once():
        nonlocal active_env_count
        active_env_count = 0
        active_sphere_counts.clear()
        total_loss = torch.zeros(B, device=device, dtype=torch.float32)
        for b in range(B):
            t0 = time.perf_counter()
            subject_positions = torch.stack(
                [state.get_position(mesh_cache.pair_subject_objs[p])[b] for p in range(num_pairs)]
            )
            obstacle_positions = torch.stack([
                (
                    mesh_cache.pair_fixed_obstacle_pos[p]
                    if mesh_cache.pair_obstacle_is_fixed[p]
                    else state.get_position(mesh_cache.pair_obstacle_objs[p])[b].detach()
                )
                for p in range(num_pairs)
            ])
            phases["01_gather_positions"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            fixed_obstacle_yaws = mesh_cache.pair_fixed_obstacle_yaw
            has_any_yaw = orientations is not None or any(y != 0.0 for y in fixed_obstacle_yaws)
            if has_any_yaw:
                ori_b = orientations[b] if orientations is not None else {}
                subject_yaws = torch.tensor(
                    [
                        (
                            ori_b.get(mesh_cache.pair_subject_objs[p], 0.0)
                            if mesh_cache.pair_subject_applies_yaw[p]
                            else 0.0
                        )
                        for p in range(num_pairs)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                obstacle_yaws = torch.tensor(
                    [ori_b.get(mesh_cache.pair_obstacle_objs[p], fixed_obstacle_yaws[p]) for p in range(num_pairs)],
                    dtype=torch.float32,
                    device=device,
                )
            phases["02_yaw_tensors"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            margins = mesh_cache.pair_max_radius + clearance_m
            s_bbox_min = mesh_cache.pair_subject_bbox_min[:, b, :]
            s_bbox_max = mesh_cache.pair_subject_bbox_max[:, b, :]
            o_bbox_min = mesh_cache.pair_obstacle_bbox_min[:, b, :]
            o_bbox_max = mesh_cache.pair_obstacle_bbox_max[:, b, :]
            if has_any_yaw:
                subject_bbox_yaws = torch.tensor(
                    [
                        0.0 if mesh_cache.pair_subject_bbox_includes_yaw[p] else subject_yaws[p].item()
                        for p in range(num_pairs)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                obstacle_bbox_yaws = torch.tensor(
                    [
                        0.0 if mesh_cache.pair_obstacle_bbox_includes_yaw[p] else obstacle_yaws[p].item()
                        for p in range(num_pairs)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                s_bbox_min, s_bbox_max = _rotate_bbox_extents(s_bbox_min, s_bbox_max, subject_bbox_yaws)
                o_bbox_min, o_bbox_max = _rotate_bbox_extents(o_bbox_min, o_bbox_max, obstacle_bbox_yaws)
            subject_min = subject_positions + s_bbox_min
            subject_max = subject_positions + s_bbox_max
            obstacle_min = obstacle_positions + o_bbox_min
            obstacle_max = obstacle_positions + o_bbox_max
            sep_subject = (subject_min - margins.unsqueeze(1)) > obstacle_max
            sep_obstacle = (obstacle_min - margins.unsqueeze(1)) > subject_max
            separated = sep_subject.any(dim=1) | sep_obstacle.any(dim=1)
            active_pair = ~separated
            _ = bool(active_pair.any())  # same sync as production early-continue
            phases["03_aabb_filter_with_item_sync"] += time.perf_counter() - t0

            if not active_pair.any():
                continue
            active_env_count += 1

            t0 = time.perf_counter()
            offsets = subject_positions - obstacle_positions
            sphere_active_mask = active_pair[mesh_cache.sphere_pair_id]
            active_idx = sphere_active_mask.nonzero(as_tuple=True)[0]
            active_sphere_pair_id = mesh_cache.sphere_pair_id[active_idx]
            local_centers = mesh_cache.all_centers_local[active_idx]
            if has_any_yaw:
                net_yaws = (subject_yaws - obstacle_yaws)[active_sphere_pair_id]
                local_centers = rotate_points_by_yaw_batch(local_centers, net_yaws)
                pair_offsets = offsets[active_sphere_pair_id]
                obs_yaws = obstacle_yaws[active_sphere_pair_id]
                rotated_offsets = rotate_points_by_yaw_batch(pair_offsets, -obs_yaws)
                active_centers = local_centers + rotated_offsets
            else:
                active_centers = local_centers + offsets[active_sphere_pair_id]
            active_radii = mesh_cache.all_radii[active_idx]
            active_mesh_idx = mesh_cache.sphere_mesh_idx[active_idx].contiguous()
            phases["04_gather_active_spheres"] += time.perf_counter() - t0
            active_sphere_counts.append(int(active_centers.shape[0]))

            t0 = time.perf_counter()
            active_mesh_indices_wp = wp.from_torch(active_mesh_idx, dtype=wp.int32)
            sdf_values = multi_mesh_sdf(active_centers, mesh_cache.mesh_id_array, active_mesh_indices_wp)
            mesh_manager.warn_sdf_sentinel(sdf_values)
            sdf_values = clamp_sdf_sentinel(sdf_values)
            phases["05_sdf_kernel"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            penetration = torch.relu(active_radii + clearance_m - sdf_values)
            pair_sum = torch.zeros(num_pairs, device=device, dtype=penetration.dtype)
            pair_sum.index_add_(0, active_sphere_pair_id, penetration)
            pair_mean = pair_sum / mesh_cache.pair_sphere_count
            active_pair_idx = active_pair.nonzero(as_tuple=True)[0]
            total_loss[b] = total_loss[b] + slope * pair_mean[active_pair_idx].sum()
            phases["06_reduce"] += time.perf_counter() - t0
        return total_loss

    # Warm + timed with CUDA sync around whole call
    run_once()
    _sync(device)
    phases.clear()
    active_env_count = 0
    t0 = time.perf_counter()
    run_once()
    _sync(device)
    total_ms = (time.perf_counter() - t0) * 1e3
    out = {k: v * 1e3 for k, v in phases.items()}
    out["total_instrumented_ms"] = total_ms
    out["active_envs"] = active_env_count
    out["avg_active_spheres"] = float(sum(active_sphere_counts) / max(1, len(active_sphere_counts)))
    out["aabb_never_filters"] = active_env_count == B
    return out


def _profile_sync_hotspots(state, mesh_cache, orientations, device) -> dict[str, float]:
    """Measure common sync/overhead patterns used in mesh loss."""
    import torch

    import warp as wp

    from isaaclab_arena.relations.warp_sdf_kernels import multi_mesh_sdf

    results = {}
    num_pairs = mesh_cache.num_pairs
    b = 0

    def build_positions():
        subject_positions = torch.stack(
            [state.get_position(mesh_cache.pair_subject_objs[p])[b] for p in range(num_pairs)]
        )
        obstacle_positions = torch.stack([
            (
                mesh_cache.pair_fixed_obstacle_pos[p]
                if mesh_cache.pair_obstacle_is_fixed[p]
                else state.get_position(mesh_cache.pair_obstacle_objs[p])[b].detach()
            )
            for p in range(num_pairs)
        ])
        return subject_positions, obstacle_positions

    results["build_pair_positions_ms"] = _timed(build_positions, device, repeats=20)

    # .item() sync storm for yaw bbox flags
    def yaw_item_storm():
        subject_yaws = torch.zeros(num_pairs, device=device)
        for p in range(num_pairs):
            _ = subject_yaws[p].item()

    results["yaw_item_loop_ms"] = _timed(yaw_item_storm, device, repeats=10)

    # torch.tensor list construction (as in mesh loss yaw path)
    ori_b = orientations[b] if orientations else {}

    def tensor_from_list():
        torch.tensor(
            [ori_b.get(mesh_cache.pair_subject_objs[p], 0.0) for p in range(num_pairs)],
            dtype=torch.float32,
            device=device,
        )

    results["torch_tensor_from_python_list_ms"] = _timed(tensor_from_list, device, repeats=20)

    # SDF for all active spheres of env 0 (worst case: all pairs active)
    subject_positions, obstacle_positions = build_positions()
    offsets = subject_positions - obstacle_positions
    active_centers = mesh_cache.all_centers_local + offsets[mesh_cache.sphere_pair_id]
    active_mesh_idx = mesh_cache.sphere_mesh_idx.contiguous()

    def all_spheres_sdf():
        idx_wp = wp.from_torch(active_mesh_idx, dtype=wp.int32)
        return multi_mesh_sdf(active_centers.contiguous(), mesh_cache.mesh_id_array, idx_wp)

    all_spheres_sdf()
    results["env0_all_spheres_sdf_ms"] = _timed(all_spheres_sdf, device, repeats=10)
    results["env0_sphere_count"] = int(mesh_cache.total_spheres)

    # wp.from_torch alone
    def from_torch_only():
        return wp.from_torch(active_mesh_idx, dtype=wp.int32)

    results["wp_from_torch_mesh_idx_ms"] = _timed(from_torch_only, device, repeats=20)

    # sentinel check sync
    sdf = all_spheres_sdf()

    def sentinel_any():
        return bool((sdf >= 1.0e5).any())

    results["sdf_sentinel_any_ms"] = _timed(sentinel_any, device, repeats=20)

    # Batch all envs in one SDF (hypothetical parallelization)
    B = state.batch_size
    # For each pair, get (B,3) subject/obstacle and expand spheres
    all_query = []
    all_mesh_idx = []
    for p in range(num_pairs):
        subj = state.get_position(mesh_cache.pair_subject_objs[p])
        if mesh_cache.pair_obstacle_is_fixed[p]:
            obs = mesh_cache.pair_fixed_obstacle_pos[p].unsqueeze(0).expand(B, 3)
        else:
            obs = state.get_position(mesh_cache.pair_obstacle_objs[p]).detach()
        off = subj - obs  # (B,3)
        mask = mesh_cache.sphere_pair_id == p
        centers = mesh_cache.all_centers_local[mask]  # (S,3)
        q = centers.unsqueeze(0) + off.unsqueeze(1)  # (B,S,3)
        all_query.append(q.reshape(-1, 3))
        all_mesh_idx.append(mesh_cache.sphere_mesh_idx[mask].repeat(B))
    query_all = torch.cat(all_query, dim=0).contiguous()
    mesh_idx_all = torch.cat(all_mesh_idx, dim=0).contiguous()

    def batched_all_envs_sdf():
        idx_wp = wp.from_torch(mesh_idx_all, dtype=wp.int32)
        return multi_mesh_sdf(query_all, mesh_cache.mesh_id_array, idx_wp)

    batched_all_envs_sdf()
    results["batched_all_envs_sdf_ms"] = _timed(batched_all_envs_sdf, device, repeats=5)
    results["batched_query_count"] = int(query_all.shape[0])
    results["serial_envs_estimated_sdf_ms"] = results["env0_all_spheres_sdf_ms"] * B
    return results


def main() -> None:
    from isaaclab_arena_environments.cli import get_isaaclab_arena_environments_cli_parser

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    parser.add_argument("--candidates", type=int, default=50, help="Candidate batch size (production-like: 50).")
    parser.add_argument("--profile_iters", type=int, default=50, help="Solver iterations for wall-time estimate.")
    parser.add_argument(
        "--mesh_impl",
        choices=("batched", "serial"),
        default="batched",
        help="Mesh no-overlap implementation used by the solver and mesh-term timing.",
    )
    args = parser.parse_args()
    placement_seed = args.placement_seed if getattr(args, "placement_seed", None) is not None else 42
    yaml_path = args.env_graph_spec_yaml

    with SimulationAppContext(args):
        result = _run_profile(yaml_path, placement_seed, args.candidates, args.profile_iters, args.mesh_impl)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
