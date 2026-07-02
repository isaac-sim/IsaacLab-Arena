# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false

"""Wall-clock benchmarks for RelationSolver and ObjectPlacer (sim-free dummy scenes)."""

from __future__ import annotations

import importlib.util
import json
import statistics
import time
import torch
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.bounding_box_helpers import assign_variants_for_envs, build_per_env_bounding_boxes
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On, get_anchor_objects
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

CollisionModeName = Literal["bbox", "mesh"]


def mesh_collision_available() -> bool:
    """True when the collision_mode module and Warp are importable."""
    if importlib.util.find_spec("isaaclab_arena.relations.collision_mode") is None:
        return False
    if importlib.util.find_spec("warp") is None:
        return False
    try:
        import warp as wp

        wp.init()
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class BenchmarkScenario:
    """One benchmark configuration."""

    name: str
    """Scenario label in printed results."""

    num_objects: int
    """Total scene objects, including one anchor table."""

    num_envs: int
    """Independent layout batch size passed to solve()."""

    max_iters: int = 600
    """Adam iteration cap per solve."""

    collision_mode: CollisionModeName = "bbox"
    """Collision backend: ``bbox`` or ``mesh``."""

    num_spheres: int = 30
    """Sphere count per object in mesh mode."""

    placement_seed: int = 0
    """RNG seed for reproducible initial positions."""

    max_placement_attempts: int = 1
    """Candidate layouts solved per ObjectPlacer.place() call."""

    warmup_runs: int = 1
    """Untimed solves before measurement."""

    timed_runs: int = 3
    """Timed solves; reported solve_ms is the median."""


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Timing result for one scenario run."""

    scenario_name: str
    """Scenario label from the benchmark configuration."""

    collision_mode: str
    """Collision backend used for this run."""

    num_objects: int
    """Total objects in the scene."""

    num_envs: int
    """Batch size."""

    num_optimizable: int
    """Movable objects (num_objects minus anchors)."""

    device: str
    """Torch device used by the solver (``cpu`` or ``cuda``)."""

    solve_ms: float
    """Median RelationSolver.solve() wall time in milliseconds."""

    place_ms: float
    """Median ObjectPlacer.place() wall time in milliseconds."""

    iters: int
    """Optimizer steps from the last timed solve."""

    overlap_pairs: int
    """No-overlap pairs scored in that solve."""

    ms_per_iter: float
    """solve_ms / iters."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Preset scenarios (object count and batch size increase together)."""
    return (
        BenchmarkScenario(name="small", num_objects=3, num_envs=1),
        BenchmarkScenario(name="medium", num_objects=6, num_envs=8),
        BenchmarkScenario(name="large", num_objects=10, num_envs=32),
    )


def object_count_sweep(
    *,
    num_envs: int = 8,
    counts: tuple[int, ...] = (3, 5, 6, 10),
    collision_mode: CollisionModeName = "bbox",
    max_iters: int = 600,
) -> tuple[BenchmarkScenario, ...]:
    """Hold batch size fixed; vary object count."""
    return tuple(
        BenchmarkScenario(
            name=f"objs_{count}",
            num_objects=count,
            num_envs=num_envs,
            max_iters=max_iters,
            collision_mode=collision_mode,
        )
        for count in counts
    )


def env_count_sweep(
    *,
    num_objects: int = 6,
    env_counts: tuple[int, ...] = (1, 8, 32),
    collision_mode: CollisionModeName = "bbox",
    max_iters: int = 600,
) -> tuple[BenchmarkScenario, ...]:
    """Hold object count fixed; vary batch size."""
    return tuple(
        BenchmarkScenario(
            name=f"envs_{count}",
            num_objects=num_objects,
            num_envs=count,
            max_iters=max_iters,
            collision_mode=collision_mode,
        )
        for count in env_counts
    )


def build_clutter_scene(num_objects: int, collision_mode: CollisionModeName = "bbox") -> list[DummyObject]:
    """Anchor table plus ``num_objects - 1`` boxes with On(table) relations."""
    if collision_mode == "mesh":
        return _build_mesh_clutter_scene(num_objects)
    return _build_bbox_clutter_scene(num_objects)


def _build_bbox_clutter_scene(num_objects: int) -> list[DummyObject]:
    assert num_objects >= 2, f"need at least anchor + one box, got {num_objects}"

    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 0.1)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    box_bbox = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.12, 0.12, 0.12))
    boxes = []
    for idx in range(num_objects - 1):
        box = DummyObject(name=f"box_{idx}", bounding_box=box_bbox)
        box.add_relation(On(table, clearance_m=0.01))
        boxes.append(box)

    return [table, *boxes]


def _build_mesh_clutter_scene(num_objects: int) -> list[DummyObject]:
    import trimesh

    assert num_objects >= 2, f"need at least anchor + one box, got {num_objects}"

    table_mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.1))
    table = DummyObject(
        name="table",
        bounding_box=AxisAlignedBoundingBox(min_point=(-0.5, -0.5, -0.05), max_point=(0.5, 0.5, 0.05)),
        collision_mesh=table_mesh,
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    box_mesh = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
    box_bbox = AxisAlignedBoundingBox(min_point=(-0.06, -0.06, -0.06), max_point=(0.06, 0.06, 0.06))
    boxes = []
    for idx in range(num_objects - 1):
        box = DummyObject(name=f"box_{idx}", bounding_box=box_bbox, collision_mesh=box_mesh)
        box.add_relation(On(table, clearance_m=0.01))
        boxes.append(box)

    return [table, *boxes]


def make_solver_params(scenario: BenchmarkScenario) -> RelationSolverParams:
    """Build RelationSolverParams for ``scenario.collision_mode``."""
    if scenario.collision_mode == "mesh":
        assert mesh_collision_available(), "mesh collision_mode requires the collision_mode module and Warp"
        from isaaclab_arena.relations.collision_mode import CollisionMode

        return RelationSolverParams(
            max_iters=scenario.max_iters,
            verbose=False,
            profile=False,
            save_position_history=False,
            collision_mode=CollisionMode.MESH,
            num_spheres=scenario.num_spheres,
        )

    return RelationSolverParams(
        max_iters=scenario.max_iters,
        verbose=False,
        profile=False,
        save_position_history=False,
    )


def make_placer_params(scenario: BenchmarkScenario) -> ObjectPlacerParams:
    """ObjectPlacer params aligned with ``scenario``."""
    return ObjectPlacerParams(
        solver_params=make_solver_params(scenario),
        placement_seed=scenario.placement_seed,
        max_placement_attempts=scenario.max_placement_attempts,
        apply_positions_to_objects=False,
        verbose=False,
    )


def _sample_child_origin(
    parent_min: float,
    parent_max: float,
    child_min: float,
    child_max: float,
    generator: torch.Generator,
) -> float:
    low = parent_min - child_min
    high = parent_max - child_max
    if low >= high:
        return float((parent_min + parent_max) / 2.0)
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


def _initial_positions_for_env(
    objects: list[DummyObject],
    anchor_objects: set[DummyObject],
    env_bboxes: dict[DummyObject, AxisAlignedBoundingBox],
    generator: torch.Generator,
) -> dict[DummyObject, tuple[float, float, float]]:
    anchor = next(iter(anchor_objects))
    anchor_pose = anchor.get_initial_pose()
    assert anchor_pose is not None
    anchor_bbox = env_bboxes[anchor].translated(anchor_pose.position_xyz)

    positions: dict[DummyObject, tuple[float, float, float]] = {}
    for obj in objects:
        if obj in anchor_objects:
            positions[obj] = anchor_pose.position_xyz
            continue

        on_relation = next(r for r in obj.get_relations() if isinstance(r, On))
        parent = on_relation.parent
        if parent in anchor_objects:
            parent_bbox = anchor_bbox
        else:
            parent_pos = positions[parent]
            parent_bbox = env_bboxes[parent].translated(parent_pos)

        child_bbox = env_bboxes[obj]
        x = _sample_child_origin(
            float(parent_bbox.min_point[0, 0]),
            float(parent_bbox.max_point[0, 0]),
            float(child_bbox.min_point[0, 0]),
            float(child_bbox.max_point[0, 0]),
            generator,
        )
        y = _sample_child_origin(
            float(parent_bbox.min_point[0, 1]),
            float(parent_bbox.max_point[0, 1]),
            float(child_bbox.min_point[0, 1]),
            float(child_bbox.max_point[0, 1]),
            generator,
        )
        z = float(parent_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])
        positions[obj] = (x, y, z)

    return positions


def build_solve_inputs(
    objects: list[DummyObject],
    num_envs: int,
    seed: int,
) -> tuple[list[dict[DummyObject, tuple[float, float, float]]], dict[DummyObject, AxisAlignedBoundingBox]]:
    """Random On(table) seeds and per-candidate bboxes for a batched solve."""
    anchor_objects = set(get_anchor_objects(objects))
    assert len(anchor_objects) == 1

    assign_variants_for_envs(objects, num_envs, placement_seed=seed)
    env_bboxes = build_per_env_bounding_boxes(objects, num_envs)
    per_env_bboxes = env_bboxes.get_bounding_boxes_for_all_envs()
    candidate_bboxes = env_bboxes.get_bounding_boxes_for_solver_candidates(1)

    initial_positions: list[dict[DummyObject, tuple[float, float, float]]] = []
    generator = torch.Generator().manual_seed(seed)
    for env_idx in range(num_envs):
        generator.manual_seed(seed + env_idx)
        initial_positions.append(
            _initial_positions_for_env(objects, anchor_objects, per_env_bboxes[env_idx], generator)
        )

    return initial_positions, candidate_bboxes


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) if samples else 0.0


def _time_solver_solve(
    solver: RelationSolver,
    objects: list[DummyObject],
    initial_positions: list[dict[DummyObject, tuple[float, float, float]]],
    candidate_bboxes: dict[DummyObject, AxisAlignedBoundingBox],
) -> float:
    _sync_cuda()
    start = time.perf_counter()
    solver.solve(objects, initial_positions, env_bboxes=candidate_bboxes)
    _sync_cuda()
    return (time.perf_counter() - start) * 1e3


def run_solver_benchmark(scenario: BenchmarkScenario) -> BenchmarkMeasurement:
    """Time RelationSolver.solve() on a dummy clutter scene."""
    objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
    solver = RelationSolver(params=make_solver_params(scenario))
    initial_positions, candidate_bboxes = build_solve_inputs(objects, scenario.num_envs, scenario.placement_seed)

    for _ in range(scenario.warmup_runs):
        _time_solver_solve(solver, objects, initial_positions, candidate_bboxes)

    timed_ms = [
        _time_solver_solve(solver, objects, initial_positions, candidate_bboxes) for _ in range(scenario.timed_runs)
    ]
    solve_ms = _median_ms(timed_ms)

    iters = len(solver.last_loss_history)
    ms_per_iter = solve_ms / iters if iters > 0 else 0.0

    return BenchmarkMeasurement(
        scenario_name=scenario.name,
        collision_mode=scenario.collision_mode,
        num_objects=scenario.num_objects,
        num_envs=scenario.num_envs,
        num_optimizable=scenario.num_objects - 1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        solve_ms=solve_ms,
        place_ms=0.0,
        iters=iters,
        overlap_pairs=solver.last_no_overlap_pair_count,
        ms_per_iter=ms_per_iter,
    )


def run_placer_benchmark(scenario: BenchmarkScenario) -> BenchmarkMeasurement:
    """Time ObjectPlacer.place() end-to-end on the same clutter scene."""
    objects = build_clutter_scene(scenario.num_objects, scenario.collision_mode)
    placer = ObjectPlacer(params=make_placer_params(scenario))

    for _ in range(scenario.warmup_runs):
        _sync_cuda()
        placer.place(objects=objects, num_envs=scenario.num_envs)
        _sync_cuda()

    timed_ms = []
    for _ in range(scenario.timed_runs):
        _sync_cuda()
        start = time.perf_counter()
        placer.place(objects=objects, num_envs=scenario.num_envs)
        _sync_cuda()
        timed_ms.append((time.perf_counter() - start) * 1e3)

    place_ms = _median_ms(timed_ms)
    iters = len(placer.last_loss_history)

    return BenchmarkMeasurement(
        scenario_name=scenario.name,
        collision_mode=scenario.collision_mode,
        num_objects=scenario.num_objects,
        num_envs=scenario.num_envs,
        num_optimizable=scenario.num_objects - 1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        solve_ms=0.0,
        place_ms=place_ms,
        iters=iters,
        overlap_pairs=placer.last_no_overlap_pair_count,
        ms_per_iter=0.0,
    )


def run_benchmarks(
    scenarios: tuple[BenchmarkScenario, ...],
    *,
    include_placer: bool = True,
) -> list[BenchmarkMeasurement]:
    """Run solver benchmarks; optionally add matching placer timing per scenario."""
    results: list[BenchmarkMeasurement] = []
    for scenario in scenarios:
        solver_row = run_solver_benchmark(scenario)
        if not include_placer:
            results.append(solver_row)
            continue
        placer_row = run_placer_benchmark(scenario)
        results.append(
            BenchmarkMeasurement(
                scenario_name=solver_row.scenario_name,
                collision_mode=solver_row.collision_mode,
                num_objects=solver_row.num_objects,
                num_envs=solver_row.num_envs,
                num_optimizable=solver_row.num_optimizable,
                device=solver_row.device,
                solve_ms=solver_row.solve_ms,
                place_ms=placer_row.place_ms,
                iters=solver_row.iters,
                overlap_pairs=solver_row.overlap_pairs,
                ms_per_iter=solver_row.ms_per_iter,
            )
        )
    return results


def scenarios_for_modes(
    base_scenarios: tuple[BenchmarkScenario, ...],
    collision_modes: tuple[CollisionModeName, ...],
) -> tuple[BenchmarkScenario, ...]:
    """Expand each base scenario across collision modes (e.g. bbox vs mesh)."""
    expanded: list[BenchmarkScenario] = []
    for scenario in base_scenarios:
        for mode in collision_modes:
            name = scenario.name if len(collision_modes) == 1 else f"{scenario.name}_{mode}"
            expanded.append(replace(scenario, name=name, collision_mode=mode))
    return tuple(expanded)


def format_results_table(rows: list[BenchmarkMeasurement]) -> str:
    """Render measurements as a fixed-width text table."""
    header = (
        f"{'scenario':<14} {'mode':<5} {'objects':>7} {'envs':>5} {'device':<5} "
        f"{'solve_ms':>10} {'place_ms':>10} {'iters':>5} {'pairs':>5} {'ms/iter':>8}"
    )
    lines = [
        "solve_ms = median RelationSolver.solve() wall time; place_ms = median ObjectPlacer.place() wall time.",
        "ms/iter = solve_ms / iters. pairs = no-overlap pairs scored. Times exclude warmup runs.",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row.scenario_name:<14} {row.collision_mode:<5} {row.num_objects:>7} {row.num_envs:>5} "
            f"{row.device:<5} {row.solve_ms:>9.1f}  {row.place_ms:>9.1f}  "
            f"{row.iters:>5} {row.overlap_pairs:>5} {row.ms_per_iter:>8.2f}"
        )
    return "\n".join(lines)


def write_results_json(path: str | Path, rows: list[BenchmarkMeasurement]) -> None:
    """Write measurements as a JSON list."""
    payload = [row.to_dict() for row in rows]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
