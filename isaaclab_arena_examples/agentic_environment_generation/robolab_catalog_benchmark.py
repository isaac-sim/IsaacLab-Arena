# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark agentic env generation against RoboLab catalog prompts.

Runs each Arena prompt from ``isaaclab_arena_environments/robolab/task_catalog.md`` through
the generation pipeline (intent generation → link/``to_arena_env`` → relation solve), records
failures per stage, and compares successful linked YAMLs to ground-truth RoboLab specs.

Usage::

    python isaaclab_arena_examples/agentic_environment_generation/robolab_catalog_benchmark.py \\
        --headless --out_dir isaaclab_arena_environments/agent_generated/robolab_benchmark
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import traceback
import yaml
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from isaaclab_arena.agentic_environment_generation.spec_io import write_env_graph_specs
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

DEFAULT_CATALOG_PATH = Path("isaaclab_arena_environments/robolab/task_catalog.md")
DEFAULT_ROBOLAB_DIR = Path("isaaclab_arena_environments/robolab")
DEFAULT_REPORT_DIR = Path("isaaclab_arena_environments/agent_generated/robolab_benchmark")

PipelineStage = Literal[
    "generate_intent",
    "link_to_arena_env",
    "relation_solve",
    "yaml_validation",
]


@dataclass
class CatalogCase:
    """One benchmark row from the RoboLab task catalog."""

    ground_truth_yaml: str
    prompt: str
    robolab_task_name: str
    robolab_task_description: str
    catalog_object_count: int | None
    catalog_subtask_count: int | None

    @property
    def case_id(self) -> str:
        stem = Path(self.ground_truth_yaml).stem
        return f"{stem}__{self.robolab_task_name}"


@dataclass
class StageTimings:
    """Wall-clock seconds per pipeline stage for one case."""

    generate_intent_s: float | None = None
    link_to_arena_env_s: float | None = None
    relation_solve_s: float | None = None
    yaml_validation_s: float | None = None
    total_s: float | None = None


@dataclass
class YamlValidationResult:
    """Checks comparing a generated linked YAML to ground truth."""

    passed: bool
    every_object_anchored: bool
    unanchored_objects: list[str] = field(default_factory=list)
    task_count_match: bool = False
    expected_task_count: int = 0
    actual_task_count: int = 0
    task_refs_match: bool = False
    task_ref_mismatches: list[dict[str, Any]] = field(default_factory=list)
    object_count_within_tolerance: bool = False
    expected_object_count: int = 0
    actual_object_count: int = 0
    missing_objects: list[str] = field(default_factory=list)
    extra_objects: list[str] = field(default_factory=list)
    relation_solve_had_fallbacks: bool | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class BenchmarkCaseResult:
    """Outcome for one catalog prompt."""

    case: CatalogCase
    status: Literal["success", "failed"]
    failed_stage: PipelineStage | None = None
    logs: str = ""
    linked_yaml_path: str | None = None
    validation: YamlValidationResult | None = None
    stage_timings_s: StageTimings = field(default_factory=StageTimings)


@dataclass
class StageTimingSummary:
    """Aggregate wall-clock timing across all cases (seconds)."""

    total_s: float = 0.0
    mean_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    count: int = 0


@dataclass
class BenchmarkReport:
    """Aggregate benchmark output."""

    started_at: str
    finished_at: str
    catalog_path: str
    robolab_dir: str
    out_dir: str
    total_cases: int
    succeeded: int
    failed: int
    stage_timing_summary_s: dict[str, StageTimingSummary] = field(default_factory=dict)
    results: list[BenchmarkCaseResult] = field(default_factory=list)


def add_robolab_benchmark_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register RoboLab catalog benchmark CLI arguments."""
    group = parser.add_argument_group("RoboLab Catalog Benchmark")
    group.add_argument(
        "--catalog-path",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to robolab/task_catalog.md (default: %(default)s).",
    )
    group.add_argument(
        "--robolab-dir",
        type=Path,
        default=DEFAULT_ROBOLAB_DIR,
        help="Directory containing ground-truth *_linked.yaml files (default: %(default)s).",
    )
    group.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON report path (default: <out_dir>/benchmark_report.json).",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model id (default: agent's built-in default).",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2).",
    )
    group.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for generated YAMLs and the JSON report (default: %(default)s).",
    )
    group.add_argument(
        "--skip-relation-solve",
        action="store_true",
        default=False,
        help="Stop after link/to_arena_env and skip relation solving (for debugging).",
    )
    group.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Run at most this many catalog cases (default: all).",
    )


def _parse_int_cell(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def _extract_linked_yaml(cell: str) -> str | None:
    match = re.search(r"\[([^\]]+_linked\.yaml)\]", cell)
    return match.group(1) if match else None


def parse_robolab_catalog(catalog_path: Path) -> list[CatalogCase]:
    """Parse benchmark cases from the RoboLab task catalog markdown table."""
    text = catalog_path.read_text(encoding="utf-8")
    cases: list[CatalogCase] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 8:
            continue
        spec_cell, prompt_cell = parts[1], parts[2]
        ground_truth_yaml = _extract_linked_yaml(spec_cell)
        prompt = prompt_cell.strip()
        if ground_truth_yaml is None or not prompt:
            continue
        cases.append(
            CatalogCase(
                ground_truth_yaml=ground_truth_yaml,
                prompt=prompt,
                robolab_task_name=parts[5],
                robolab_task_description=parts[6],
                catalog_object_count=_parse_int_cell(parts[4]),
                catalog_subtask_count=_parse_int_cell(parts[7]),
            )
        )
    assert cases, f"No benchmark cases with Arena prompts found in {catalog_path}"
    return cases


def _capture_logs(fn, *args, **kwargs) -> tuple[Any, str]:
    """Run ``fn`` and return ``(result, combined_stdout_stderr)``."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def _node_id_to_registry_name(spec: dict[str, Any]) -> dict[str, str]:
    """Map graph node ids to their registry asset names."""
    return {node["id"]: node["name"] for node in spec.get("nodes", [])}


def _object_registry_names(spec: dict[str, Any]) -> set[str]:
    """Return registry names for object-type nodes."""
    return {node["name"] for node in spec.get("nodes", []) if node.get("type") == "object"}


def _resolve_node_ref_to_registry_name(ref: str, id_to_name: dict[str, str]) -> str:
    """Resolve a task or constraint node reference to a registry asset name."""
    return id_to_name.get(ref, ref)


def _registry_names_for_node_ids(node_ids: set[str], id_to_name: dict[str, str]) -> set[str]:
    return {_resolve_node_ref_to_registry_name(node_id, id_to_name) for node_id in node_ids}


def _initial_state_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    state_specs = spec.get("state_specs") or []
    if not state_specs:
        return None
    for state in state_specs:
        if state.get("id") == "state_initial":
            return state
    return state_specs[0]


def _objects_reachable_from_anchors(
    state_spec: dict[str, Any], object_ids: set[str], id_to_name: dict[str, str]
) -> tuple[bool, list[str]]:
    """Return whether every object node is connected to an ``is_anchor`` node."""
    constraints = state_spec.get("spatial_constraints") or []
    anchor_ids = {c["subject"] for c in constraints if c.get("kind") == "is_anchor"}
    if not anchor_ids:
        return False, sorted(_registry_names_for_node_ids(object_ids, id_to_name))

    adjacency: dict[str, set[str]] = {}
    for constraint in constraints:
        subject = constraint.get("subject")
        if not subject:
            continue
        adjacency.setdefault(subject, set())
        reference = constraint.get("reference")
        if reference:
            adjacency.setdefault(reference, set())
            adjacency[subject].add(reference)
            adjacency[reference].add(subject)

    reachable: set[str] = set()
    stack = list(anchor_ids)
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        for neighbor in adjacency.get(node_id, ()):
            if neighbor not in reachable:
                stack.append(neighbor)

    unanchored_ids = {object_id for object_id in object_ids if object_id not in reachable}
    unanchored = sorted(_registry_names_for_node_ids(unanchored_ids, id_to_name))
    return not unanchored_ids, unanchored


def _task_object_refs(task: dict[str, Any], node_ids: set[str]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for key, value in (task.get("params") or {}).items():
        if isinstance(value, str) and value in node_ids:
            refs[key] = value
        elif isinstance(value, list):
            list_refs = [item for item in value if isinstance(item, str) and item in node_ids]
            if list_refs:
                refs[key] = list_refs
    return refs


def _task_object_refs_by_registry_name(
    task: dict[str, Any], node_ids: set[str], id_to_name: dict[str, str]
) -> dict[str, Any]:
    """Return task params that reference graph nodes, keyed by param and valued by registry name."""
    refs: dict[str, Any] = {}
    for key, value in _task_object_refs(task, node_ids).items():
        if isinstance(value, str):
            refs[key] = _resolve_node_ref_to_registry_name(value, id_to_name)
        else:
            refs[key] = [_resolve_node_ref_to_registry_name(item, id_to_name) for item in value]
    return refs


def validate_linked_yaml(generated: dict[str, Any], ground_truth: dict[str, Any]) -> YamlValidationResult:
    """Compare a generated linked YAML dict against ground truth."""
    issues: list[str] = []
    generated_id_to_name = _node_id_to_registry_name(generated)
    ground_truth_id_to_name = _node_id_to_registry_name(ground_truth)
    generated_object_ids = {node["id"] for node in generated.get("nodes", []) if node.get("type") == "object"}
    generated_objects = _object_registry_names(generated)
    ground_truth_objects = _object_registry_names(ground_truth)
    all_node_ids = set(generated_id_to_name)

    initial_state = _initial_state_spec(generated)
    if initial_state is None:
        every_object_anchored = False
        unanchored_objects = sorted(generated_objects)
        issues.append("generated spec has no initial state_spec")
    else:
        every_object_anchored, unanchored_objects = _objects_reachable_from_anchors(
            initial_state, generated_object_ids, generated_id_to_name
        )
        if not every_object_anchored:
            issues.append(f"objects not reachable from anchors: {unanchored_objects}")

    expected_task_count = len(ground_truth.get("tasks") or [])
    actual_task_count = len(generated.get("tasks") or [])
    task_count_match = expected_task_count == actual_task_count
    if not task_count_match:
        issues.append(f"task count mismatch: expected {expected_task_count}, got {actual_task_count}")

    task_ref_mismatches: list[dict[str, Any]] = []
    generated_tasks = generated.get("tasks") or []
    ground_truth_tasks = ground_truth.get("tasks") or []
    ground_truth_node_ids = set(ground_truth_id_to_name)
    task_refs_match = task_count_match
    for index, (generated_task, ground_truth_task) in enumerate(zip(generated_tasks, ground_truth_tasks)):
        generated_refs = _task_object_refs_by_registry_name(generated_task, all_node_ids, generated_id_to_name)
        ground_truth_refs = _task_object_refs_by_registry_name(
            ground_truth_task, ground_truth_node_ids, ground_truth_id_to_name
        )
        if generated_task.get("kind") != ground_truth_task.get("kind"):
            task_refs_match = False
            task_ref_mismatches.append({
                "task_index": index,
                "field": "kind",
                "expected": ground_truth_task.get("kind"),
                "actual": generated_task.get("kind"),
            })
        for key, expected_value in ground_truth_refs.items():
            actual_value = generated_refs.get(key)
            if actual_value != expected_value:
                task_refs_match = False
                task_ref_mismatches.append({
                    "task_index": index,
                    "field": key,
                    "expected": expected_value,
                    "actual": actual_value,
                })
        for key in generated_refs:
            if key not in ground_truth_refs:
                task_refs_match = False
                task_ref_mismatches.append({
                    "task_index": index,
                    "field": key,
                    "expected": None,
                    "actual": generated_refs[key],
                })
    if task_ref_mismatches:
        issues.append(f"task reference mismatches: {task_ref_mismatches}")

    expected_object_count = len(ground_truth_objects)
    actual_object_count = len(generated_objects)
    object_count_within_tolerance = abs(actual_object_count - expected_object_count) <= 1
    if not object_count_within_tolerance:
        issues.append(f"object count out of tolerance: expected {expected_object_count} ±1, got {actual_object_count}")

    missing_objects = sorted(ground_truth_objects - generated_objects)
    extra_objects = sorted(generated_objects - ground_truth_objects)
    if missing_objects:
        issues.append(f"missing object registry names vs ground truth: {missing_objects}")
    if extra_objects:
        issues.append(f"extra object registry names vs ground truth: {extra_objects}")

    passed = every_object_anchored and task_count_match and task_refs_match and object_count_within_tolerance
    return YamlValidationResult(
        passed=passed,
        every_object_anchored=every_object_anchored,
        unanchored_objects=unanchored_objects,
        task_count_match=task_count_match,
        expected_task_count=expected_task_count,
        actual_task_count=actual_task_count,
        task_refs_match=task_refs_match,
        task_ref_mismatches=task_ref_mismatches,
        object_count_within_tolerance=object_count_within_tolerance,
        expected_object_count=expected_object_count,
        actual_object_count=actual_object_count,
        missing_objects=missing_objects,
        extra_objects=extra_objects,
        issues=issues,
    )


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"Expected YAML mapping in {path}"
    return data


def _generate_initial_graph_spec(args_cli: argparse.Namespace, prompt: str):
    """Run the LLM agent and compile its intent into an initial graph spec."""
    from isaaclab_arena.agentic_environment_generation.environment_generation_agent import (
        EnvironmentGenerationAgent,
        build_asset_catalogue,
        build_relation_catalogue,
        build_task_catalogue,
    )
    from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler

    asset_catalog = build_asset_catalogue()
    relation_catalog = build_relation_catalogue()
    task_catalog = build_task_catalogue()
    agent_kwargs = {"model": args_cli.model} if args_cli.model else {}
    agent = EnvironmentGenerationAgent(**agent_kwargs)
    intent_spec, _raw_response = agent.generate_spec(
        prompt,
        asset_catalog=asset_catalog,
        relation_catalog=relation_catalog,
        task_catalog=task_catalog,
        temperature=args_cli.temperature,
    )
    return IntentCompiler().compile(intent_spec)


def _fail_result(
    case: CatalogCase,
    stage: PipelineStage,
    exc: BaseException,
    log_parts: list[str],
    linked_yaml_path: Path | None,
    stage_timings: StageTimings,
) -> BenchmarkCaseResult:
    log_parts.append(f"[{stage}] {type(exc).__name__}: {exc}")
    log_parts.append(traceback.format_exc())
    return BenchmarkCaseResult(
        case=case,
        status="failed",
        failed_stage=stage,
        logs="\n".join(log_parts),
        linked_yaml_path=str(linked_yaml_path) if linked_yaml_path else None,
        stage_timings_s=stage_timings,
    )


def _summarize_stage_timings(results: list[BenchmarkCaseResult]) -> dict[str, StageTimingSummary]:
    """Aggregate per-stage timings across completed cases."""
    stage_values: dict[str, list[float]] = {
        "generate_intent": [],
        "link_to_arena_env": [],
        "relation_solve": [],
        "yaml_validation": [],
        "total": [],
    }
    for result in results:
        timings = result.stage_timings_s
        for stage, values in (
            ("generate_intent", timings.generate_intent_s),
            ("link_to_arena_env", timings.link_to_arena_env_s),
            ("relation_solve", timings.relation_solve_s),
            ("yaml_validation", timings.yaml_validation_s),
            ("total", timings.total_s),
        ):
            if values is not None:
                stage_values[stage].append(values)

    summary: dict[str, StageTimingSummary] = {}
    for stage, values in stage_values.items():
        if not values:
            continue
        summary[stage] = StageTimingSummary(
            total_s=sum(values),
            mean_s=sum(values) / len(values),
            min_s=min(values),
            max_s=max(values),
            count=len(values),
        )
    return summary


def _link_and_build_arena_env(initial_graph_spec, case_out_dir: Path):
    """Link an initial graph spec, write YAML, and materialize an ``IsaacLabArenaEnvironment``."""
    linked_spec = initial_graph_spec.link()
    _, linked_yaml_path = write_env_graph_specs(initial_graph_spec, linked_spec, case_out_dir)
    arena_env = linked_spec.to_arena_env()
    return linked_spec, linked_yaml_path, arena_env


def run_benchmark_case(
    case: CatalogCase,
    args_cli: argparse.Namespace,
    robolab_dir: Path,
    out_dir: Path,
) -> BenchmarkCaseResult:
    """Run all pipeline stages for one catalog case."""
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement

    print(f"\n[benchmark] case: {case.case_id}", flush=True)
    print(f"[benchmark] prompt: {case.prompt!r}", flush=True)

    log_parts: list[str] = []
    stage_timings = StageTimings()
    case_started = time.perf_counter()
    ground_truth_path = robolab_dir / case.ground_truth_yaml
    assert ground_truth_path.is_file(), f"Ground-truth YAML not found: {ground_truth_path}"

    case_out_dir = out_dir / case.case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    linked_yaml_path: Path | None = None
    relation_solve_had_fallbacks: bool | None = None

    stage_started = time.perf_counter()
    try:
        initial_spec, stage_logs = _capture_logs(_generate_initial_graph_spec, args_cli, case.prompt)
        log_parts.append(stage_logs)
    except Exception as exc:
        stage_timings.generate_intent_s = time.perf_counter() - stage_started
        stage_timings.total_s = time.perf_counter() - case_started
        return _fail_result(case, "generate_intent", exc, log_parts, None, stage_timings)
    stage_timings.generate_intent_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    try:
        linked_spec, linked_yaml_path, arena_env = _link_and_build_arena_env(initial_spec, case_out_dir)
        log_parts.append(f"[link_to_arena_env] wrote linked YAML to {linked_yaml_path}")
    except Exception as exc:
        stage_timings.link_to_arena_env_s = time.perf_counter() - stage_started
        stage_timings.total_s = time.perf_counter() - case_started
        return _fail_result(case, "link_to_arena_env", exc, log_parts, linked_yaml_path, stage_timings)
    stage_timings.link_to_arena_env_s = time.perf_counter() - stage_started

    if not args_cli.skip_relation_solve:
        stage_started = time.perf_counter()
        try:
            objects_with_relations = arena_env.scene.get_objects_with_relations()
            _, stage_logs = _capture_logs(
                solve_and_apply_relation_placement,
                objects_with_relations,
                args_cli.num_envs,
                args_cli.placement_seed,
                args_cli.resolve_on_reset,
                args_cli.random_yaw_init,
            )
            log_parts.append(stage_logs)
            relation_solve_had_fallbacks = "accepted best-loss fallback layouts" in stage_logs
        except Exception as exc:
            stage_timings.relation_solve_s = time.perf_counter() - stage_started
            stage_timings.total_s = time.perf_counter() - case_started
            return _fail_result(case, "relation_solve", exc, log_parts, linked_yaml_path, stage_timings)
        stage_timings.relation_solve_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    try:
        generated_dict = linked_spec.to_dict()
        ground_truth_dict = _load_yaml_dict(ground_truth_path)
        validation = validate_linked_yaml(generated_dict, ground_truth_dict)
        validation.relation_solve_had_fallbacks = relation_solve_had_fallbacks
        if relation_solve_had_fallbacks:
            validation.issues.append("relation solver accepted fallback layouts")
            validation.passed = False
        log_parts.append("[yaml_validation] " + ("passed" if validation.passed else "failed"))
        for issue in validation.issues:
            log_parts.append(f"  - {issue}")
    except Exception as exc:
        stage_timings.yaml_validation_s = time.perf_counter() - stage_started
        stage_timings.total_s = time.perf_counter() - case_started
        return _fail_result(case, "yaml_validation", exc, log_parts, linked_yaml_path, stage_timings)
    stage_timings.yaml_validation_s = time.perf_counter() - stage_started
    stage_timings.total_s = time.perf_counter() - case_started

    status: Literal["success", "failed"] = "success" if validation.passed else "failed"
    return BenchmarkCaseResult(
        case=case,
        status=status,
        failed_stage=None if status == "success" else "yaml_validation",
        logs="\n".join(log_parts),
        linked_yaml_path=str(linked_yaml_path),
        validation=validation,
        stage_timings_s=stage_timings,
    )


def _serialize_report(report: BenchmarkReport) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {key: convert(value) for key, value in asdict(obj).items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    return convert(report)


def write_report(report: BenchmarkReport, report_path: Path) -> None:
    """Write the benchmark report as JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_report(report), handle, indent=2, sort_keys=False)
        handle.write("\n")


def _print_summary(report: BenchmarkReport) -> None:
    print("\n=== RoboLab catalog benchmark summary ===", flush=True)
    print(f"Cases: {report.total_cases}  succeeded: {report.succeeded}  failed: {report.failed}", flush=True)
    if report.stage_timing_summary_s:
        print("Stage timings (s):", flush=True)
        for stage, stats in report.stage_timing_summary_s.items():
            print(
                f"  {stage:20s} mean={stats.mean_s:7.2f}  total={stats.total_s:8.2f}"
                f"  min={stats.min_s:7.2f}  max={stats.max_s:7.2f}  n={stats.count}",
                flush=True,
            )
    for result in report.results:
        case = result.case
        timings = result.stage_timings_s
        timing_bits = []
        if timings.generate_intent_s is not None:
            timing_bits.append(f"intent={timings.generate_intent_s:.1f}s")
        if timings.link_to_arena_env_s is not None:
            timing_bits.append(f"link={timings.link_to_arena_env_s:.1f}s")
        if timings.relation_solve_s is not None:
            timing_bits.append(f"relations={timings.relation_solve_s:.1f}s")
        if timings.yaml_validation_s is not None:
            timing_bits.append(f"validate={timings.yaml_validation_s:.1f}s")
        if timings.total_s is not None:
            timing_bits.append(f"total={timings.total_s:.1f}s")
        timing_suffix = f"  ({', '.join(timing_bits)})" if timing_bits else ""
        if result.status == "success":
            print(f"  OK   {case.case_id}{timing_suffix}", flush=True)
            continue
        stage = result.failed_stage or "unknown"
        print(f"  FAIL {case.case_id}  stage={stage}{timing_suffix}", flush=True)
        if result.validation and result.validation.issues:
            for issue in result.validation.issues[:3]:
                print(f"       - {issue}", flush=True)


def run_benchmark(
    cases: list[CatalogCase],
    args_cli: argparse.Namespace,
    robolab_dir: Path,
    out_dir: Path,
    report_path: Path,
) -> BenchmarkReport:
    """Run the benchmark for all cases inside a single SimulationApp session."""
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[BenchmarkCaseResult] = []
    with SimulationAppContext(args_cli):
        for case in cases:
            results.append(run_benchmark_case(case, args_cli, robolab_dir, out_dir))

        finished_at = datetime.now(timezone.utc).isoformat()
        succeeded = sum(1 for result in results if result.status == "success")
        report = BenchmarkReport(
            started_at=started_at,
            finished_at=finished_at,
            catalog_path=str(args_cli.catalog_path.resolve()),
            robolab_dir=str(robolab_dir),
            out_dir=str(out_dir),
            total_cases=len(cases),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            stage_timing_summary_s=_summarize_stage_timings(results),
            results=results,
        )
        write_report(report, report_path)
        _print_summary(report)
        print(f"\n[benchmark] report → {report_path}", flush=True)
    return report


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_robolab_benchmark_cli_args(parser)
    args_cli = parser.parse_args()

    catalog_path = args_cli.catalog_path.resolve()
    robolab_dir = args_cli.robolab_dir.resolve()
    out_dir = args_cli.out_dir.resolve()
    report_path = (args_cli.report_path or out_dir / "benchmark_report.json").resolve()

    cases = parse_robolab_catalog(catalog_path)
    if args_cli.max_cases is not None:
        cases = cases[: args_cli.max_cases]

    report = run_benchmark(cases, args_cli, robolab_dir, out_dir, report_path)
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
