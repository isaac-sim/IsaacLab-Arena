# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run construction, summaries, and report writers for relation benchmarks."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkTarget,
    CollisionModeName,
    DiagnosticTopic,
)
from isaaclab_arena.relations.benchmark.provenance import collect_software_metadata


@dataclass(frozen=True)
class _ScalingWorkload:
    """Parameters that must match before comparing one scaling axis."""

    target: BenchmarkTarget
    worker_id: str
    collision_mode: CollisionModeName
    num_objects: int | None
    num_envs: int | None
    graph_spec_path: str | None
    include_robot: bool | None
    max_iters: int
    convergence_threshold: float
    num_spheres: int
    placement_seed: int
    max_placement_attempts: int
    warmup_runs: int
    timed_runs: int
    final_loss_threshold: float
    min_valid_layout_rate: float

    @classmethod
    def from_measurement(
        cls,
        measurement: BenchmarkMeasurement,
        scaling_axis: Literal["batch", "objects"],
    ) -> _ScalingWorkload:
        """Build the workload identity for a measurement."""
        return cls(
            target=measurement.target,
            worker_id=measurement.worker_id,
            collision_mode=measurement.collision_mode,
            num_objects=None if scaling_axis == "objects" else measurement.num_objects,
            num_envs=None if scaling_axis == "batch" else measurement.num_envs,
            graph_spec_path=measurement.graph_spec_path,
            include_robot=measurement.include_robot,
            max_iters=measurement.max_iters,
            convergence_threshold=measurement.convergence_threshold,
            num_spheres=measurement.num_spheres,
            placement_seed=measurement.placement_seed,
            max_placement_attempts=measurement.max_placement_attempts,
            warmup_runs=measurement.warmup_runs,
            timed_runs=measurement.timed_runs,
            final_loss_threshold=measurement.final_loss_threshold,
            min_valid_layout_rate=measurement.min_valid_layout_rate,
        )


def _batch_ms(measurement: BenchmarkMeasurement) -> float | None:
    return {
        "solver": measurement.solve_ms,
        "placer": measurement.place_ms,
        "environment": measurement.bring_up_ms,
    }[measurement.target]


def _iterations_per_second(measurement: BenchmarkMeasurement) -> float | None:
    return measurement.solver_iterations_per_second


def _layouts_per_second(measurement: BenchmarkMeasurement) -> float | None:
    return measurement.throughput_envs_per_second


def requested_scenario_ids(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
) -> tuple[str, ...]:
    """Return every expected result ID in execution order."""
    ids = tuple(scenario.scenario_id(target) for scenario in scenarios for target in targets)
    if len(ids) != len(set(ids)):
        raise ValueError("requested benchmark scenario IDs must be unique")
    return ids


def build_run(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]] | None = None,
    worker_exit_codes: dict[str, int] | None = None,
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build the canonical run envelope."""
    expected = requested_scenario_ids(scenarios, targets)
    exit_codes = worker_exit_codes or {"local": 0}
    if worker_assignments is None:
        worker_id = next(iter(exit_codes)) if len(exit_codes) == 1 else "local"
        worker_assignments = {worker_id: expected}
    return BenchmarkRun(
        requested_scenario_ids=expected,
        results=tuple(results),
        worker_assignments=worker_assignments,
        worker_exit_codes=exit_codes,
        software=collect_software_metadata(),
        worker_errors=worker_errors or {},
    )


def build_distributed_run(
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]],
    worker_exit_codes: dict[str, int],
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build a run whose requested IDs are already worker-qualified."""
    expected = tuple(scenario_id for ids in worker_assignments.values() for scenario_id in ids)
    return BenchmarkRun(
        requested_scenario_ids=expected,
        results=tuple(results),
        worker_assignments=worker_assignments,
        worker_exit_codes=worker_exit_codes,
        software=collect_software_metadata(),
        worker_errors=worker_errors or {},
    )


def format_results_table(results: list[BenchmarkMeasurement]) -> str:
    """Render a compact text report."""
    header = (
        f"{'scenario':<28} {'worker':<10} {'target':<11} {'status':<7} {'mode':<5} {'objects':>7} "
        f"{'batch':>5} {'batch_ms':>10} {'iter/s':>9} {'iters':>7} "
        f"{'layouts/s':>10} {'agg layouts/s':>13} "
        f"{'loss':>10} {'valid':>7}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.scenario_name:<28} {result.worker_id:<10} {result.target:<11} {result.status:<7} "
            f"{result.collision_mode:<5} {result.num_objects:>7} {result.num_envs:>5} "
            f"{_format_number(_batch_ms(result)):>10} {_format_number(_iterations_per_second(result)):>9} "
            f"{_format_iterations(result.iterations):>7} "
            f"{_format_number(result.throughput_envs_per_second):>10} "
            f"{_format_number(result.aggregate_throughput_envs_per_second):>13} "
            f"{_format_number(result.final_loss):>10} {_format_number(result.valid_layout_rate):>7}"
        )
        if result.error:
            lines.append(f"  error: {result.error}")
    return "\n".join(lines)


def format_scaling_summary(results: list[BenchmarkMeasurement]) -> str:
    """Summarize independent batch-size and object-count sweeps."""
    sections = [
        _format_batch_scaling(results),
        _format_object_scaling(results),
    ]
    return "\n\n".join(section for section in sections if section)


def format_diagnostic_markdown(run: BenchmarkRun) -> str:
    """Render a question-driven solver diagnostic page."""
    results = list(run.results)
    sections = [
        "# Relation Solver Diagnostic Benchmark",
        _format_diagnostic_context(run),
        _format_diagnostic_methodology(),
        _format_batch_diagnostic(results),
        _format_object_diagnostic(results),
        _format_background_diagnostic(results),
        _format_robot_diagnostic(results),
        _format_scene_diagnostic(results),
    ]
    return "\n\n".join(section for section in sections if section)


def write_batch_scaling_svg(path: str | Path, results: list[BenchmarkMeasurement]) -> None:
    """Plot optimization rate against batch size."""
    _write_scaling_svg(
        path,
        _diagnostic_results(results, "batchification"),
        x_value=lambda result: result.num_envs,
        title="Solver Iteration Rate vs. Batch Size",
        x_label="Batch Size",
    )


def write_object_scaling_svg(path: str | Path, results: list[BenchmarkMeasurement]) -> None:
    """Plot optimization rate against movable object count."""
    _write_scaling_svg(
        path,
        _diagnostic_results(results, "object-complexity"),
        x_value=lambda result: result.num_objects - 1,
        title="Solver Iteration Rate vs. Number of Movable Objects",
        x_label="Number of Movable Objects",
    )


def write_robot_scaling_svg(path: str | Path, results: list[BenchmarkMeasurement]) -> None:
    """Plot robot impact over movable object count."""
    series = (
        (
            "BBox / No Robot",
            "#0072B2",
            "9 6",
            lambda result: result.collision_mode == "bbox" and not result.include_robot,
        ),
        (
            "BBox / Robot",
            "#0072B2",
            None,
            lambda result: result.collision_mode == "bbox" and bool(result.include_robot),
        ),
        (
            "Mesh / No Robot",
            "#D55E00",
            "9 6",
            lambda result: result.collision_mode == "mesh" and not result.include_robot,
        ),
        (
            "Mesh / Robot",
            "#D55E00",
            None,
            lambda result: result.collision_mode == "mesh" and bool(result.include_robot),
        ),
    )
    _write_scaling_svg(
        path,
        _diagnostic_results(results, "robot-impact"),
        x_value=_movable_object_count,
        title="Robot Impact on Solver Iteration Rate",
        x_label="Number of Movable Objects",
        series=series,
    )


def _write_scaling_svg(
    path: str | Path,
    measurements: list[BenchmarkMeasurement],
    *,
    x_value: Callable[[BenchmarkMeasurement], int],
    title: str,
    x_label: str,
    series: tuple[tuple[str, str, str | None, Callable[[BenchmarkMeasurement], bool]], ...] | None = None,
) -> None:
    """Plot BBox and mesh iteration rates over one integer-valued axis."""
    x_values = sorted({x_value(result) for result in measurements})
    rates = [result.solver_iterations_per_second for result in measurements]
    finite_rates = [rate for rate in rates if rate is not None]
    assert x_values and finite_rates, "scaling diagnostic results must include iteration rates"
    width, height = 760, 440
    left, right, top, bottom = 85, 30, 90, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(finite_rates) * 1.1

    def x_position(value: int) -> float:
        index = x_values.index(value)
        return left + index * plot_width / max(len(x_values) - 1, 1)

    def y_position(rate: float) -> float:
        return top + plot_height * (1.0 - rate / y_max)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>text{font-family:sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}"
            ".grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:3}</style>"
        ),
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="18">{title}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        lines.extend([
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>',
            f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12">{value:.0f}</text>',
        ])
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
    ])
    for value in x_values:
        x = x_position(value)
        lines.append(f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-size="12">{value}</text>')
    if series is None:
        series = (
            ("BBox", "#2f6fdd", None, lambda result: result.collision_mode == "bbox"),
            ("Mesh", "#d4513f", None, lambda result: result.collision_mode == "mesh"),
        )
    for label, color, dash, predicate in series:
        mode_results = sorted(
            (
                result
                for result in measurements
                if predicate(result) and result.solver_iterations_per_second is not None
            ),
            key=x_value,
        )
        points = " ".join(
            f"{x_position(x_value(result)):.1f},{y_position(result.solver_iterations_per_second):.1f}"
            for result in mode_results
            if result.solver_iterations_per_second is not None
        )
        dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
        lines.append(f'<polyline class="series" stroke="{color}"{dash_attribute} points="{points}"/>')
        for result in mode_results:
            assert result.solver_iterations_per_second is not None
            x = x_position(x_value(result))
            y = y_position(result.solver_iterations_per_second)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
    for index, (label, color, dash, _predicate) in enumerate(series):
        legend_x = width - 360 + (index % 2) * 180
        legend_y = 47 + (index // 2) * 22
        lines.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}"'
            f' stroke="{color}" stroke-width="3"'
            + ("/>" if dash is None else f' stroke-dasharray="{dash}"/>'),
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-size="12">{label}</text>',
        ])
    lines.extend([
        f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="14">{x_label}</text>',
        (
            f'<text x="18" y="{height / 2}" text-anchor="middle" font-size="14"'
            f' transform="rotate(-90 18 {height / 2})">Iterations per Second (iter/s)</text>'
        ),
        "</svg>",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_droid_scene_snapshot(path: str | Path) -> None:
    """Save one environment from the maintained homogeneous DROID rendering."""
    from PIL import Image

    source = Path(__file__).resolve().parents[3] / "docs" / "images" / "same_objects_different_layouts.gif"
    with Image.open(source) as image:
        image.seek(0)
        width, height = image.size
        image.crop((0, 0, width // 2, height // 2)).convert("RGB").save(path)


def write_lightwheel_kitchen_snapshot(path: str | Path) -> None:
    """Save the rendered object-count kitchen scene."""
    source = Path(__file__).resolve().parents[3] / "docs" / "images" / "kitchen_background_collision.png"
    Path(path).write_bytes(source.read_bytes())


def write_droid_kitchen_snapshot(path: str | Path) -> None:
    """Save one rendered Lightwheel kitchen scene with DROID."""
    from PIL import Image

    source = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "images"
        / "agentic_environment_generation"
        / "droid_kitchen_pnp_pi.gif"
    )
    with Image.open(source) as image:
        image.seek(0)
        image.convert("RGB").save(path)


def _diagnostic_results(
    results: list[BenchmarkMeasurement],
    topic: DiagnosticTopic,
) -> list[BenchmarkMeasurement]:
    return [result for result in results if result.diagnostic_topic == topic]


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([header, divider, *("| " + " | ".join(row) + " |" for row in rows)])


def _format_diagnostic_context(run: BenchmarkRun) -> str:
    devices = {
        (
            result.device.name or "CPU",
            result.device.compute_capability or "-",
            result.device.total_memory_bytes,
        )
        for result in run.results
    }
    device_text = ", ".join(
        f"{name} (compute {capability}, {_format_gib(memory)} GiB)" for name, capability, memory in sorted(devices)
    )
    dirty = "dirty" if run.software.git_dirty else "clean"
    commit = run.software.git_commit[:12] if run.software.git_commit else "unavailable"
    return (
        "## Hardware and software\n"
        f"- Device: {device_text or 'unavailable'}\n"
        f"- Git: `{commit}` ({dirty})\n"
        f"- Python: {run.software.python_version}; PyTorch: {run.software.pytorch_version}; "
        f"CUDA: {run.software.cuda_version or 'unavailable'}"
    )


def _format_diagnostic_methodology() -> str:
    return (
        "## Methodology\n"
        "- **iterations/s** times only the Adam optimization loop and is the primary solver-kernel metric.\n"
        "- **solve/place ms** includes setup and result processing; **layouts/s** is completed batch throughput.\n"
        "- **AABB/mesh pairs** count directed no-overlap checks in the final timed run.\n"
        "- A row is failed when loss is non-finite, the configured loss gate fails, or placement validity is too low."
    )


def _format_batch_diagnostic(results: list[BenchmarkMeasurement]) -> str:
    measurements = sorted(
        _diagnostic_results(results, "batchification"),
        key=lambda result: (result.collision_mode, result.num_envs),
    )
    if not measurements:
        return ""
    rows = [
        (
            _collision_mode_label(result.collision_mode),
            str(result.num_envs),
            _format_number(result.solver_iterations_per_second),
            _format_number(result.solve_ms),
            _format_number(result.throughput_envs_per_second),
            _format_iterations(result.iterations),
            _diagnostic_outcome(result),
        )
        for result in measurements
    ]
    findings = []
    for mode in ("bbox", "mesh"):
        mode_results = [result for result in measurements if result.collision_mode == mode]
        if len(mode_results) > 1:
            first, last = mode_results[0], mode_results[-1]
            findings.append(
                f"{_collision_mode_label(mode)}: batch {first.num_envs}→{last.num_envs} changed optimization rate"
                f" {_format_ratio(last.solver_iterations_per_second, first.solver_iterations_per_second)} and layout"
                f" throughput {_format_ratio(last.throughput_envs_per_second, first.throughput_envs_per_second)}."
            )
            successful_batches = [result.num_envs for result in mode_results if result.status == "ok"]
            failed_batches = [result.num_envs for result in mode_results if result.status == "failed"]
            if failed_batches:
                highest_success = max(successful_batches) if successful_batches else None
                findings.append(
                    f"{_collision_mode_label(mode)}: highest batch passing the strict loss gate was"
                    f" {highest_success if highest_success is not None else 'none'}; failures started at"
                    f" batch {min(failed_batches)}."
                )
    return (
        "## Performance Changes for BBox/Mesh with Batch Size on a Real Table-Top Scene\n"
        "- Environment: maintained DROID Maple-table homogeneous placement scene\n"
        "- Workload: orange, ketchup bottle, soup can, spoon, and sugar box\n"
        "- Experiment: measure optimization iterations/s and total solve time at batches 1, 8, 32, 128, and 256\n"
        "- Each batch entry is one independently solved layout; no physics stepping is measured\n\n"
        "### Environment snapshot\n"
        "![One DROID homogeneous Maple-table environment](table_scene.png)\n\n"
        "**Question:** Does batching improve solver-kernel and completed-layout throughput in each collision mode?\n\n"
        + _format_findings(findings)
        + "\n\n![BBox and mesh iterations/s by batch size](batch_scaling.svg)\n\n"
        + _markdown_table(
            ("Mode", "Batch", "iterations/s", "solve ms", "layouts/s", "iterations", "Outcome"),
            rows,
        )
    )


def _format_object_diagnostic(results: list[BenchmarkMeasurement]) -> str:
    measurements = sorted(
        _diagnostic_results(results, "object-complexity"),
        key=lambda result: (result.collision_mode, result.num_objects),
    )
    if not measurements:
        return ""
    rows = []
    findings = []
    for mode in ("bbox", "mesh"):
        mode_results = [result for result in measurements if result.collision_mode == mode]
        baseline = mode_results[0].solver_iterations_per_second if mode_results else None
        for result in mode_results:
            rows.append((
                _collision_mode_label(mode),
                str(result.num_objects - 1),
                _format_number(result.solver_iterations_per_second),
                _format_ratio(result.solver_iterations_per_second, baseline),
                _format_number(result.solve_ms),
                str(_pair_count(result)),
            ))
        if len(mode_results) > 1:
            first, last = mode_results[0], mode_results[-1]
            findings.append(
                f"{_collision_mode_label(mode)}: {first.num_objects - 1}→{last.num_objects - 1} movable objects"
                " changed optimization rate"
                f" {_format_ratio(last.solver_iterations_per_second, first.solver_iterations_per_second)} and"
                f" collision pairs {_pair_count(first)}→{_pair_count(last)}."
            )
    return (
        "## Performance Changes for BBox/Mesh with Object Count in the Lightwheel RoboCasa Kitchen\n"
        "### Set Up\n"
        "- Lightwheel RoboCasa kitchen right counter\n"
        "- Nested subsets of registered YCB, HOPE, and RoboLab objects\n"
        "- BBox/Mesh collision modes\n"
        "- On relation only\n"
        "- Batch size: 1\n"
        "- Movable object counts: 1, 2, 3, 5, 10, 15, and 20\n\n"
        "### Command to Run\n"
        "```bash\n"
        "/isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\\n"
        "  --suite diagnostic \\\n"
        "  --diagnostic-topic object-complexity \\\n"
        "  --max-iters 200 \\\n"
        "  --output-dir /workspaces/solver-object-count-kitchen\n"
        "```\n\n"
        "### Scene Screenshot\n"
        "![Lightwheel RoboCasa kitchen counter](kitchen_scene.png)\n\n"
        "### Result\n"
        "**Question:** How does adding optimized objects affect iteration speed and collision work at batch 1?\n\n"
        + _format_findings(findings)
        + "\n\n![BBox and Mesh iterations/s by movable object count](object_scaling.svg)\n\n"
        + _markdown_table(
            ("Mode", "Movable Objects", "iterations/s", "vs 1 object", "solve ms", "collision pairs"),
            rows,
        )
    )


def _format_background_diagnostic(results: list[BenchmarkMeasurement]) -> str:
    measurements = sorted(
        _diagnostic_results(results, "background-collision"),
        key=lambda result: (result.collision_mode, result.background_treatment),
    )
    if not measurements:
        return ""
    rows = [
        (
            _collision_mode_label(result.collision_mode),
            result.background_treatment,
            str(result.background_object_count or 0),
            str(result.aabb_pair_count or 0),
            str(result.mesh_pair_count or 0),
            _format_number(result.solver_iterations_per_second),
            _format_number(result.final_loss),
            _diagnostic_outcome(result),
        )
        for result in measurements
    ]
    findings = []
    for mode in ("bbox", "mesh"):
        mode_results = [result for result in measurements if result.collision_mode == mode]
        baseline = next((result for result in mode_results if result.background_treatment == "none"), None)
        treated = next((result for result in mode_results if result.background_treatment != "none"), None)
        if baseline is not None and treated is not None:
            findings.append(
                f"{_collision_mode_label(mode)}: adding {treated.background_treatment} background added"
                f" {_pair_count(treated) - _pair_count(baseline)} directed collision pairs and changed optimization"
                f" rate {_format_ratio(treated.solver_iterations_per_second, baseline.solver_iterations_per_second)}."
            )
    return (
        "## Background collision impact\n"
        "**Question:** What optimization cost is introduced by fixed AABB or mesh background geometry?\n\n"
        + _format_findings(findings)
        + "\n\n"
        + _markdown_table(
            ("Mode", "Background", "Objects", "AABB pairs", "mesh pairs", "iterations/s", "loss", "Outcome"),
            rows,
        )
    )


def _format_robot_diagnostic(results: list[BenchmarkMeasurement]) -> str:
    measurements = sorted(
        _diagnostic_results(results, "robot-impact"),
        key=lambda result: (result.collision_mode, _movable_object_count(result), bool(result.include_robot)),
    )
    if not measurements:
        return ""
    baselines = {
        (result.collision_mode, _movable_object_count(result)): result
        for result in measurements
        if not result.include_robot
    }
    rows = []
    for result in measurements:
        baseline = baselines.get((result.collision_mode, _movable_object_count(result)))
        rows.append((
            _collision_mode_label(result.collision_mode),
            str(_movable_object_count(result)),
            "Yes" if result.include_robot else "No",
            _format_number(result.solver_iterations_per_second),
            _format_ratio(
                result.solver_iterations_per_second,
                baseline.solver_iterations_per_second if baseline is not None else None,
            ),
            _format_number(result.solve_ms),
            str(_pair_count(result)),
        ))
    findings = []
    for mode in ("bbox", "mesh"):
        mode_results = [result for result in measurements if result.collision_mode == mode]
        comparable_counts = {
            _movable_object_count(result)
            for result in mode_results
            if result.include_robot and (mode, _movable_object_count(result)) in baselines
        }
        if not comparable_counts:
            continue
        largest_count = max(comparable_counts)
        without_robot = baselines[(mode, largest_count)]
        with_robot = next(
            result for result in mode_results if result.include_robot and _movable_object_count(result) == largest_count
        )
        if without_robot is not None and with_robot is not None:
            findings.append(
                f"{_collision_mode_label(mode)} at {largest_count} objects: adding the robot changed iteration rate"
                f" {_format_ratio(with_robot.solver_iterations_per_second, without_robot.solver_iterations_per_second)},"
                f" solve time {_format_ratio(with_robot.solve_ms, without_robot.solve_ms)}, and collision pairs"
                f" {_pair_count(without_robot)}→{_pair_count(with_robot)}."
            )
    return (
        "## Performance Changes with a Robot in the Lightwheel RoboCasa Kitchen\n"
        "### Set Up\n"
        "- Same kitchen, object counts, and batch size as Experiment 2\n"
        "- Compare with and without the DROID robot\n"
        "- BBox/Mesh collision modes\n"
        "- 200 solver iterations\n\n"
        "### Command to Run\n"
        "```bash\n"
        "/isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\\n"
        "  --suite diagnostic \\\n"
        "  --diagnostic-topic robot-impact \\\n"
        "  --max-iters 200 \\\n"
        "  --output-dir ~/solver-benchmark-results/exp3\n"
        "```\n\n"
        "### Scene\n"
        "![Lightwheel kitchen with DROID](robot_scene.png)\n\n"
        "### Result\n"
        "**Question:** How much solver overhead does the robot add at each object count?\n\n"
        + _format_findings(findings)
        + "\n\n![Robot impact on iterations/s](robot_scaling.svg)\n\n"
        + _markdown_table(
            ("Mode", "Objects", "Robot", "iterations/s", "vs No Robot", "solve ms", "collision pairs"),
            rows,
        )
    )


def _format_scene_diagnostic(results: list[BenchmarkMeasurement]) -> str:
    measurements = sorted(
        _diagnostic_results(results, "scene-difficulty"),
        key=lambda result: result.graph_spec_path or result.scenario_name,
    )
    if not measurements:
        return ""
    rows = []
    for result in measurements:
        attempt_count = result.num_envs * result.timed_runs
        valid_layouts = (
            f"{round(result.valid_layout_rate * attempt_count)}/{attempt_count}"
            if result.valid_layout_rate is not None
            else "-"
        )
        median_iterations = statistics.median(result.iterations) if result.iterations else None
        solver_time_s = result.place_ms / 1e3 if result.place_ms is not None else None
        rows.append((
            _scene_name(result),
            _format_number(solver_time_s),
            valid_layouts,
            f"{median_iterations:g}" if median_iterations is not None else "-",
        ))
    passed_count = sum(result.status == "ok" for result in measurements)
    completed = [result for result in measurements if result.place_ms is not None]
    findings = [f"{passed_count}/{len(measurements)} catalog environments passed all layout attempts."]
    capped = [
        result
        for result in measurements
        if result.iterations and all(iterations == result.max_iters for iterations in result.iterations)
    ]
    if capped:
        findings.append(
            f"{len(capped)}/{len(measurements)} environments used the full iteration budget; reaching the cap does not"
            " by itself mean that layout validation failed."
        )
    if completed:

        def place_ms(result: BenchmarkMeasurement) -> float:
            assert result.place_ms is not None
            return result.place_ms

        slowest = max(completed, key=place_ms)
        assert slowest.place_ms is not None
        findings.append(
            f"{_scene_name(slowest)} had the"
            f" largest median placement time ({_format_number(slowest.place_ms / 1e3)} s)."
        )
    preamble = "\n".join((
        "## Valid-Layout Setup Time Across the Kitchen Benchmark Catalog",
        "- Workload: all 17 maintained kitchen environment graph specs, including their configured robot and objects",
        "- Batch size: 1 environment",
        "- Runs: 3, using placement seeds 0, 1, and 2",
        "- Measurement: end-to-end ObjectPlacer time; simulation stepping and asset loading are excluded",
        "",
        "### Command to Run",
        "```bash",
        "/isaac-sim/python.sh isaaclab_arena_examples/relations/relation_solver_benchmark.py \\",
        "  --suite diagnostic \\",
        "  --diagnostic-topic scene-difficulty \\",
        "  --warmup 0 \\",
        "  --repeat 3 \\",
        "  --output-dir ~/solver-benchmark-results/exp4",
        "```",
        "",
        "### Result",
        "**Question:** How long does the solver take to produce a valid layout for each kitchen catalog environment?",
        "",
        "",
    ))
    return (
        preamble
        + _format_findings(findings)
        + "\n\n"
        + _markdown_table(
            ("Environment YAML", "Median solver time (s)", "Valid layouts", "Median iterations"),
            rows,
        )
    )


def _scene_name(result: BenchmarkMeasurement) -> str:
    return Path(result.graph_spec_path).name if result.graph_spec_path else result.scenario_name


def _format_findings(findings: list[str]) -> str:
    return "**Answer:** " + " ".join(findings) if findings else "**Answer:** insufficient comparable results."


def _format_ratio(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0.0:
        return "-"
    return f"{value / baseline:.2f}×"


def _format_gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 2**30:.1f}"


def _pair_count(result: BenchmarkMeasurement) -> int:
    return (result.aabb_pair_count or 0) + (result.mesh_pair_count or 0)


def _movable_object_count(result: BenchmarkMeasurement) -> int:
    if result.asset_set_name == "lightwheel-kitchen-counter" and result.include_robot:
        return result.num_objects - 3
    return result.num_objects - 1


def _collision_mode_label(mode: str) -> str:
    return "BBox" if mode == "bbox" else "Mesh"


def _diagnostic_outcome(result: BenchmarkMeasurement) -> str:
    return result.status if result.error is None else f"{result.status}: {result.error}"


def _format_batch_scaling(results: list[BenchmarkMeasurement]) -> str:
    """Summarize batch-size scaling with object count held fixed."""
    groups: dict[_ScalingWorkload, list[BenchmarkMeasurement]] = {}
    for result in results:
        groups.setdefault(_ScalingWorkload.from_measurement(result, "batch"), []).append(result)

    lines = []
    for workload, measurements in groups.items():
        if len({measurement.num_envs for measurement in measurements}) <= 1:
            continue
        successful = [measurement for measurement in measurements if measurement.status == "ok"]
        ordered = sorted(measurements, key=lambda measurement: measurement.num_envs)
        baseline = ordered[0]
        if workload.target == "solver":
            rate = _iterations_per_second
            rate_label = "iterations/s"
            comparison_label = f"iteration_rate_vs_batch_{baseline.num_envs}"
        else:
            rate = _layouts_per_second
            rate_label = "layouts/s"
            comparison_label = f"throughput_vs_batch_{baseline.num_envs}"
        baseline_rate = rate(baseline)
        highest = max((measurement.num_envs for measurement in successful), default=None)
        failures = sorted(
            (
                measurement.num_envs,
                measurement.error or "unknown failure",
            )
            for measurement in measurements
            if measurement.status == "failed"
        )
        assert workload.num_objects is not None
        workload_description = f"objects={workload.num_objects}"
        if workload.graph_spec_path is not None:
            robot = "yes" if workload.include_robot else "no"
            workload_description += f", graph={workload.graph_spec_path}, robot={robot}"
        highest_text = "-" if highest is None else str(highest)
        points = []
        for measurement in ordered:
            measured_rate = rate(measurement)
            scale = measured_rate / baseline_rate if measured_rate is not None and baseline_rate is not None else None
            rate_text = "-" if measured_rate is None else f"{measured_rate:.3f} {rate_label}"
            batch_ms = _format_number(_batch_ms(measurement))
            layouts_per_second = ""
            if workload.target == "solver":
                layouts_per_second = f", {_format_number(measurement.throughput_envs_per_second)} layouts/s"
            scale_text = "" if scale is None else f", {scale:.2f}x"
            points.append(
                f"{measurement.num_envs}={rate_text}, {batch_ms} ms{layouts_per_second}"
                f"{scale_text}, {measurement.status}"
            )
        failure_text = "; ".join(f"{num_envs}: {reason}" for num_envs, reason in failures) or "-"
        lines.append(
            f"{workload.target}/{workload.worker_id} [{workload.collision_mode}, {workload_description}]: "
            f"{comparison_label}: "
            + "; ".join(points)
            + f"; highest successful batch={highest_text}; failures={failure_text}"
        )
    return "\n".join(["Batch-size scaling (fixed object count)", *lines]) if lines else ""


def _format_object_scaling(results: list[BenchmarkMeasurement]) -> str:
    """Summarize object-count scaling with batch size held fixed."""
    groups: dict[_ScalingWorkload, list[BenchmarkMeasurement]] = {}
    for result in results:
        groups.setdefault(_ScalingWorkload.from_measurement(result, "objects"), []).append(result)

    lines = []
    for workload, measurements in groups.items():
        if len({measurement.num_objects for measurement in measurements}) <= 1:
            continue
        ordered = sorted(measurements, key=lambda measurement: measurement.num_objects)
        if workload.target == "solver":
            metric = _iterations_per_second
            metric_label = "iterations/s"
            comparison_label = f"iteration_rate_vs_objects_{ordered[0].num_objects}"
        else:
            metric = _batch_ms
            metric_label = "ms"
            comparison_label = f"latency_vs_objects_{ordered[0].num_objects}"
        baseline_value = metric(ordered[0])
        points = []
        for measurement in ordered:
            measured_value = metric(measurement)
            scale = (
                measured_value / baseline_value if measured_value is not None and baseline_value is not None else None
            )
            metric_text = "-" if measured_value is None else f"{measured_value:.3f} {metric_label}"
            batch_ms = _format_number(_batch_ms(measurement))
            scale_text = "" if scale is None else f", {scale:.2f}x"
            points.append(f"{measurement.num_objects}={metric_text}, {batch_ms} ms{scale_text}, {measurement.status}")
        assert workload.num_envs is not None
        lines.append(
            f"{workload.target}/{workload.worker_id} [{workload.collision_mode}, batch={workload.num_envs}]: "
            f"{comparison_label}: "
            + "; ".join(points)
        )
    return "\n".join(["Object-count scaling (fixed batch size)", *lines]) if lines else ""


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_iterations(iterations: tuple[int, ...] | None) -> str:
    return "-" if not iterations else f"{statistics.median(iterations):.0f}"


def write_results_json(path: str | Path, run: BenchmarkRun) -> None:
    """Write the canonical benchmark envelope."""
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_results_csv(path: str | Path, run: BenchmarkRun) -> None:
    """Write the run's result rows as CSV."""
    rows = [result.to_dict() for result in run.results]
    fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["device"] = json.dumps(row["device"], sort_keys=True)
            writer.writerow(row)
