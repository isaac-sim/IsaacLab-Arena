# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark agentic env-gen docs prompts against their expected graph YAMLs.

Extracts the prompts and expected YAMLs from ``docs/pages/example_workflows/agentic_env_gen``
and the corresponding specs under ``isaaclab_arena_environments/``. Runs each prompt ``n`` times
against an inference endpoint and scores the generated spec with a structural compare.

Reports land under ``output/docs_prompt_benchmark/<endpoint>_<model>_<short_hash>/`` by default.

Usage::

    /isaac-sim/python.sh isaaclab_arena_examples/agentic_environment_generation/docs_prompt_benchmark.py \\
        --inference_endpoint internal --num_runs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.agentic_environment_generation.inference_backend import (
    INFERENCE_ENDPOINTS,
    resolve_inference_endpoint,
)
from isaaclab_arena.agentic_environment_generation.simready_asset_search import SimReadySearchConfig
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_ROOT = _REPO_ROOT / "output" / "docs_prompt_benchmark"
_RELATION_KINDS = ("is_anchor", "on", "next_to")


def _sanitize_model_name(model: str) -> str:
    """Make a model id safe for a single path component."""
    return model.replace("/", "-").replace(" ", "_")


def _short_commit_hash() -> str:
    """Return the current HEAD short hash, or ``unknown`` when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def benchmark_run_dir(endpoint: str, model: str, *, commit_hash: str | None = None) -> Path:
    """Return ``output/docs_prompt_benchmark/<endpoint>_<model>_<short_hash>``."""
    short_hash = commit_hash or _short_commit_hash()
    folder = f"{endpoint}_{_sanitize_model_name(model)}_{short_hash}"
    return _DEFAULT_REPORT_ROOT / folder


@dataclass(frozen=True)
class BenchmarkCase:
    """One docs workflow prompt and its reviewed expected YAML."""

    name: str
    prompt: str
    expected_yaml: Path
    enable_simready_search: bool = False


# Prompts and expected YAMLs from docs/pages/example_workflows/agentic_env_gen/*/step_1_launch_runner.rst
# and the reviewed specs linked from each workflow's index.rst.
DOC_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="kitchen_open_door",
        prompt=(
            "There is a floor and a fridge in the lightwheel_robocasa_kitchen kitchen. "
            "DROID is on the floor, next to the fridge with 0.1 meter distance and facing it. "
            "DROID opens the fridge door to the 0.2 openness threshold."
        ),
        expected_yaml=_REPO_ROOT
        / "isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml",
    ),
    BenchmarkCase(
        name="kitchen_pick_and_place",
        prompt=(
            "There is a center-right counter top and a floor in the lightwheel_robocasa_kitchen "
            "background. DROID picks up a mustard bottle on the counter top and places it in a bowl. "
            "DROID is next to the counter top and on the floor."
        ),
        expected_yaml=_REPO_ROOT
        / "isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml",
    ),
    BenchmarkCase(
        name="tabletop_pnp_homogenous_object",
        prompt=(
            "Droid picks up the banana from the maple table and places it on the plate. "
            "There are two bagels and one bowl on the table."
        ),
        expected_yaml=_REPO_ROOT / "isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml",
    ),
    BenchmarkCase(
        name="tabletop_pnp_heterogeneous_object",
        prompt=(
            "Droid picks up a fruit from the maple table and places it into the bowl on the table. "
            "Each environment should get a different fruit."
        ),
        expected_yaml=_REPO_ROOT
        / "isaaclab_arena_environments/maple_table_top/droid_pick_fruit_into_bowl_maple_table.yaml",
    ),
    BenchmarkCase(
        name="tabletop_pnp_composite_task",
        prompt=(
            "Droid picks up the pepsi can and the bean can from the maple table and places them "
            "into the mini plastic basket. There is a hammer next to the pepsi can and a tuna can "
            "on the table, and the bean can sits next to the basket."
        ),
        expected_yaml=_REPO_ROOT
        / "isaaclab_arena_environments/maple_table_top/simready_droid_pick_place_cans_hammer_maple_table.yaml",
        enable_simready_search=True,
    ),
)


@dataclass
class CompareResult:
    """Per-check outcomes for one generated vs expected pair."""

    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """One generate_spec attempt for one case."""

    case_name: str
    run_index: int
    passed: bool
    failures: list[str]
    generate_spec_runtime_s: float
    spec_inference_num_calls: int
    spec_inference_num_retries: int
    spec_inference_retry_errors: list[str]
    generation_failed: bool = False


def _object_ids(spec: ArenaEnvGraphSpec) -> set[str]:
    return {obj.id for obj in spec.objects}


def _task_referenced_ids(spec: ArenaEnvGraphSpec) -> set[str]:
    """Return every asset id named in a subtask param value."""
    refs: set[str] = set()
    for subtask in spec.task.subtasks:
        for value in subtask.params.values():
            if isinstance(value, str):
                refs.add(value)
    return refs


def _task_used_object_count(spec: ArenaEnvGraphSpec) -> int:
    """Count ordinary objects referenced by the task (distractors ignored)."""
    return len(_task_referenced_ids(spec) & _object_ids(spec))


def _object_reference_type_counts(spec: ArenaEnvGraphSpec) -> Counter[str]:
    counts: Counter[str] = Counter()
    for reference in spec.object_references or []:
        counts[reference.object_type.value] += 1
    return counts


def _relation_kind_counts(spec: ArenaEnvGraphSpec) -> Counter[str]:
    counts: Counter[str] = Counter()
    for relation in spec.relations:
        if relation.kind in _RELATION_KINDS:
            counts[relation.kind] += 1
    return counts


def compare_specs(expected: ArenaEnvGraphSpec, actual: ArenaEnvGraphSpec) -> CompareResult:
    """Score ``actual`` against ``expected`` with the docs-benchmark structural rules."""
    failures: list[str] = []

    if actual.embodiment.registry_name != expected.embodiment.registry_name:
        failures.append(
            f"embodiment registry_name {actual.embodiment.registry_name!r} != {expected.embodiment.registry_name!r}"
        )
    if actual.background.registry_name != expected.background.registry_name:
        failures.append(
            f"background registry_name {actual.background.registry_name!r} != {expected.background.registry_name!r}"
        )

    expected_refs = expected.object_references or []
    actual_refs = actual.object_references or []
    if len(actual_refs) != len(expected_refs):
        failures.append(f"object_references count {len(actual_refs)} != {len(expected_refs)}")
    expected_types = _object_reference_type_counts(expected)
    actual_types = _object_reference_type_counts(actual)
    if actual_types != expected_types:
        failures.append(f"object_reference types {dict(actual_types)} != {dict(expected_types)}")

    expected_sets = len(expected.object_sets or [])
    actual_sets = len(actual.object_sets or [])
    if actual_sets != expected_sets:
        failures.append(f"object_sets count {actual_sets} != {expected_sets}")

    expected_task_objects = _task_used_object_count(expected)
    actual_task_objects = _task_used_object_count(actual)
    if actual_task_objects != expected_task_objects:
        failures.append(f"task-used objects count {actual_task_objects} != {expected_task_objects}")

    expected_relations = _relation_kind_counts(expected)
    actual_relations = _relation_kind_counts(actual)
    for kind in _RELATION_KINDS:
        if actual_relations[kind] != expected_relations[kind]:
            failures.append(f"{kind} relation count {actual_relations[kind]} != {expected_relations[kind]}")

    if actual.task.composition != expected.task.composition:
        failures.append(f"task composition {actual.task.composition!r} != {expected.task.composition!r}")
    if len(actual.task.subtasks) != len(expected.task.subtasks):
        failures.append(f"subtask count {len(actual.task.subtasks)} != {len(expected.task.subtasks)}")
    else:
        for index, (actual_subtask, expected_subtask) in enumerate(
            zip(actual.task.subtasks, expected.task.subtasks, strict=True)
        ):
            if actual_subtask.kind != expected_subtask.kind:
                failures.append(f"subtask[{index}] kind {actual_subtask.kind!r} != {expected_subtask.kind!r}")

    return CompareResult(passed=not failures, failures=failures)


def _run_case(
    case: BenchmarkCase,
    *,
    run_index: int,
    endpoint: str,
) -> RunResult:
    expected = ArenaEnvGraphSpec.from_yaml(case.expected_yaml)
    agent = EnvironmentGenerationAgent(
        endpoint=endpoint,
        enable_simready_search=case.enable_simready_search,
        simready_config=SimReadySearchConfig() if case.enable_simready_search else None,
    )
    started = time.perf_counter()
    spec, _data = agent.generate_spec(case.prompt)
    runtime_s = time.perf_counter() - started
    stats = dict(agent.spec_inference.last_infer_stats)

    if spec is None:
        failures = list(agent.traces) or ["generate_spec returned no spec"]
        return RunResult(
            case_name=case.name,
            run_index=run_index,
            passed=False,
            failures=failures,
            generate_spec_runtime_s=runtime_s,
            spec_inference_num_calls=int(stats.get("num_calls", 0)),
            spec_inference_num_retries=int(stats.get("num_retries", 0)),
            spec_inference_retry_errors=list(stats.get("retry_errors", [])),
            generation_failed=True,
        )

    compare = compare_specs(expected, spec)
    return RunResult(
        case_name=case.name,
        run_index=run_index,
        passed=compare.passed,
        failures=compare.failures,
        generate_spec_runtime_s=runtime_s,
        spec_inference_num_calls=int(stats.get("num_calls", 0)),
        spec_inference_num_retries=int(stats.get("num_retries", 0)),
        spec_inference_retry_errors=list(stats.get("retry_errors", [])),
    )


def _format_report(
    *,
    endpoint: str,
    model: str,
    commit: str,
    num_runs: int,
    results: list[RunResult],
) -> str:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    lines: list[str] = [
        f"# Docs prompt benchmark ({endpoint})",
        "",
        f"- endpoint: {endpoint}",
        f"- model: {model}",
        f"- commit: {commit}",
        f"- runs per prompt: {num_runs}",
        f"- total attempts: {total}",
        f"- pass: {passed} ({(100.0 * passed / total) if total else 0.0:.1f}%)",
        f"- fail: {failed} ({(100.0 * failed / total) if total else 0.0:.1f}%)",
        "",
        "## Failed cases",
        "",
    ]
    failures_by_case: dict[str, list[RunResult]] = {}
    for result in results:
        if not result.passed:
            failures_by_case.setdefault(result.case_name, []).append(result)
    if not failures_by_case:
        lines.append("None.")
    else:
        for case_name, case_failures in failures_by_case.items():
            lines.append(f"### {case_name}")
            for result in case_failures:
                lines.append(f"- run {result.run_index}:")
                for failure in result.failures:
                    lines.append(f"  - {failure}")
            lines.append("")

    lines.extend(["## Per prompt+run timing and retries", ""])
    for result in results:
        lines.append(
            f"- {result.case_name} run {result.run_index}: "
            f"{'PASS' if result.passed else 'FAIL'}, "
            f"runtime={result.generate_spec_runtime_s:.2f}s, "
            f"spec_inference_calls={result.spec_inference_num_calls}, "
            f"retries_with_error={result.spec_inference_num_retries}"
        )
        for error in result.spec_inference_retry_errors:
            lines.append(f"  - retry error: {error}")

    total_runtime = sum(result.generate_spec_runtime_s for result in results)
    total_retries = sum(result.spec_inference_num_retries for result in results)
    total_calls = sum(result.spec_inference_num_calls for result in results)
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- total generate_spec runtime: {total_runtime:.2f}s",
        f"- mean generate_spec runtime: {(total_runtime / total) if total else 0.0:.2f}s",
        f"- total spec_inference calls: {total_calls}",
        f"- total spec_inference retries with error: {total_retries}",
        "",
    ])

    lines.append("## Per-case aggregates")
    lines.append("")
    by_case: dict[str, list[RunResult]] = {}
    for result in results:
        by_case.setdefault(result.case_name, []).append(result)
    for case_name, case_results in by_case.items():
        case_pass = sum(1 for result in case_results if result.passed)
        case_runtime = sum(result.generate_spec_runtime_s for result in case_results)
        case_retries = sum(result.spec_inference_num_retries for result in case_results)
        lines.append(
            f"- {case_name}: {case_pass}/{len(case_results)} pass, "
            f"runtime={case_runtime:.2f}s, retries_with_error={case_retries}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference_endpoint",
        type=str,
        default="internal",
        choices=tuple(INFERENCE_ENDPOINTS),
        help="Inference endpoint preset (default: internal).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="How many generate_spec calls to make per docs prompt (default: 3).",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of case names to run.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Directory for report.md / results.json. Default: "
            "output/docs_prompt_benchmark/<endpoint>_<model>_<short_hash>/."
        ),
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional markdown report path override (default: <out_dir>/report.md).",
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=None,
        help="Optional JSON dump path override (default: <out_dir>/results.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = resolve_inference_endpoint(args.inference_endpoint)
    out_dir = args.out_dir or benchmark_run_dir(endpoint.name, endpoint.model)
    report_path = args.report_path or (out_dir / "report.md")
    json_path = args.json_path or (out_dir / "results.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark] out_dir={out_dir}", flush=True)

    selected = DOC_BENCHMARK_CASES
    if args.cases:
        wanted = set(args.cases)
        selected = tuple(case for case in DOC_BENCHMARK_CASES if case.name in wanted)
        missing = wanted - {case.name for case in selected}
        assert not missing, f"Unknown case names: {sorted(missing)}"
    for case in selected:
        assert case.expected_yaml.is_file(), f"Missing expected YAML: {case.expected_yaml}"

    results: list[RunResult] = []
    for case in selected:
        for run_index in range(1, args.num_runs + 1):
            print(f"[benchmark] {case.name} run {run_index}/{args.num_runs}", flush=True)
            result = _run_case(case, run_index=run_index, endpoint=args.inference_endpoint)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"[benchmark] {case.name} run {run_index}: {status} "
                f"({result.generate_spec_runtime_s:.2f}s, retries={result.spec_inference_num_retries})",
                flush=True,
            )
            if result.failures:
                for failure in result.failures:
                    print(f"[benchmark]   - {failure}", flush=True)
            results.append(result)

    commit = _short_commit_hash()
    report = _format_report(
        endpoint=endpoint.name,
        model=endpoint.model,
        commit=commit,
        num_runs=args.num_runs,
        results=results,
    )
    print(report, flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] wrote report → {report_path}", flush=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": endpoint.name,
        "model": endpoint.model,
        "commit": commit,
        "out_dir": str(out_dir),
        "runs": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[benchmark] wrote json → {json_path}", flush=True)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
