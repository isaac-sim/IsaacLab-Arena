# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark time from sending a prompt until the agent returns a parseable environment spec.

Endpoint initialization and catalogue construction happen before measurements begin. A sample
includes spec inference, validation and critic retries, and prim-path resolution when the spec has
object references. It stops when ``EnvironmentGenerationAgent.generate_spec`` returns an
``ArenaEnvGraphSpec``. Percentiles include successful samples only.

Usage::

    /isaac-sim/python.sh \
        isaaclab_arena_examples/agentic_environment_generation/benchmark/benchmark_time_to_first_spec.py \
        --inference_endpoint internal --num_runs 100 \
        --output_path outputs/time_to_first_spec.json
"""

from __future__ import annotations

import argparse
import json
import time
import yaml
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from isaaclab_arena.agentic_environment_generation.catalogues import (
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.agentic_environment_generation.simready_search_interface import (
    SimReadySourceKind,
    simready_search_config_from_cli,
)
from isaaclab_arena.agentic_environment_generation.spec_io import write_env_graph_spec
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.inference.backend import resolve_inference_endpoint
from isaaclab_arena_examples.agentic_environment_generation.benchmark.latency_summary import summarize_latencies
from isaaclab_arena_examples.agentic_environment_generation.cli_runner import add_agent_inference_cli_args

DEFAULT_OUTPUT_PATH = Path("outputs/time_to_first_spec.json")
DEFAULT_CASES_PATH = Path(__file__).with_name("time_to_first_spec_cases.yaml")


@dataclass(frozen=True)
class BenchmarkCase:
    """One prompt workload and whether its expected path needs reference resolution."""

    name: str
    prompt: str
    object_references_expected: bool | None = None
    enable_simready_search: bool = False
    simready_sources: list[str] = field(default_factory=lambda: [SimReadySourceKind.ISAAC_SIM_GA.value])
    simready_s3_url: str | None = None
    simready_service_url: str | None = None
    simready_max_results_per_object: int = 1

    def __post_init__(self):
        assert self.prompt.strip(), f"benchmark case {self.name!r} must define a non-empty prompt"
        assert self.object_references_expected in (
            None,
            True,
            False,
        ), f"benchmark case {self.name!r} object_references_expected must be true, false, or omitted"
        assert isinstance(
            self.enable_simready_search, bool
        ), f"benchmark case {self.name!r} enable_simready_search must be true or false"
        valid_sources = [kind.value for kind in SimReadySourceKind]
        assert self.simready_sources and all(
            source in valid_sources for source in self.simready_sources
        ), f"benchmark case {self.name!r} simready_sources must be a non-empty subset of {valid_sources}"
        assert (
            isinstance(self.simready_max_results_per_object, int) and self.simready_max_results_per_object > 0
        ), f"benchmark case {self.name!r} simready_max_results_per_object must be a positive integer"


def load_benchmark_cases(path: Path) -> tuple[dict[str, BenchmarkCase], str]:
    """Load manually defined prompt cases and the default case name from YAML.

    Args:
        path: YAML file containing ``default_case`` and a ``cases`` mapping.

    Returns:
        Cases keyed by name and the configured default case name.
    """
    assert path.is_file(), f"benchmark cases file not found: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"benchmark cases file must contain a mapping: {path}"
    raw_cases = data.get("cases")
    assert (
        isinstance(raw_cases, dict) and raw_cases
    ), f"benchmark cases file must define a non-empty 'cases' mapping: {path}"

    case_fields = {case_field.name for case_field in fields(BenchmarkCase)} - {"name"}
    cases: dict[str, BenchmarkCase] = {}
    for name, values in raw_cases.items():
        assert isinstance(name, str) and name, f"benchmark case names must be non-empty strings: {name!r}"
        assert isinstance(values, dict), f"benchmark case {name!r} must contain a mapping"
        unexpected = set(values) - case_fields
        assert not unexpected, f"benchmark case {name!r} has unknown fields: {sorted(unexpected)}"
        prompt = values.get("prompt")
        assert isinstance(prompt, str), f"benchmark case {name!r} must define a prompt string"
        cases[name] = BenchmarkCase(name=name, **{**values, "prompt": prompt.strip()})

    default_case = data.get("default_case")
    assert (
        isinstance(default_case, str) and default_case in cases
    ), f"default_case must name one of the configured cases: {sorted(cases)}"
    return cases, default_case


@dataclass(frozen=True)
class SampleResult:
    """Result of one prompt-to-spec latency sample."""

    run_index: int
    time_to_first_spec_ms: float | None
    error: str | None = None
    generated_spec_path: str | None = None


def _run_sample(
    agent: EnvironmentGenerationAgent,
    prompt: str,
    run_index: int,
    *,
    asset_catalog: Any,
    relation_catalog: Any,
    task_catalog: Any,
    object_references_expected: bool | None = None,
    spec_output_dir: Path | None = None,
) -> SampleResult:
    error: str | None = None
    spec: ArenaEnvGraphSpec | None = None
    started_at = time.perf_counter()
    try:
        spec, _ = agent.generate_spec(
            prompt,
            asset_catalog=asset_catalog,
            relation_catalog=relation_catalog,
            task_catalog=task_catalog,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started_at) * 1e3 if spec is not None else None

    if spec is not None and object_references_expected is not None:
        has_object_references = bool(spec.object_references)
        if has_object_references != object_references_expected:
            expected = "with" if object_references_expected else "without"
            actual = "with" if has_object_references else "without"
            error = f"workload expected a spec {expected} object_references, got one {actual} them"
            latency_ms = None
    if spec is None and error is None:
        error = "; ".join(agent.traces) or "agent returned no parseable ArenaEnvGraphSpec"

    generated_spec_path = None
    if latency_ms is not None and spec_output_dir is not None:
        generated_spec_path = str(write_env_graph_spec(spec, spec_output_dir))
    return SampleResult(
        run_index=run_index,
        time_to_first_spec_ms=latency_ms,
        error=error,
        generated_spec_path=generated_spec_path,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_agent_inference_cli_args(parser, include_prompt=False)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--case",
        default=None,
        help="Case name from --cases_file (default: that file's default_case).",
    )
    workload.add_argument("--prompt", default=None, help="Custom environment-generation prompt to benchmark.")
    parser.add_argument(
        "--cases_file",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Manually defined benchmark cases (default: {DEFAULT_CASES_PATH}).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        required=True,
        help="Number of measured requests. Use at least 100 when reporting p99.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--spec_output_dir",
        type=Path,
        default=None,
        help="Optional directory for the first accepted spec, named from its env_name.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark, write sample-level results plus p50/p95/p99, and return 1 if any sample failed."""
    args = parse_args()
    assert args.num_runs > 0, f"num_runs must be positive, got {args.num_runs}"

    benchmark_case: BenchmarkCase | None = None
    if args.prompt is None:
        benchmark_cases, default_case_name = load_benchmark_cases(args.cases_file)
        case_name = args.case or default_case_name
        assert (
            case_name in benchmark_cases
        ), f"unknown benchmark case {case_name!r} in {args.cases_file}; available: {sorted(benchmark_cases)}"
        benchmark_case = benchmark_cases[case_name]
    prompt = benchmark_case.prompt if benchmark_case is not None else args.prompt
    assert prompt is not None

    endpoint = resolve_inference_endpoint(args.inference_endpoint)
    simready_config = None
    if benchmark_case is not None and benchmark_case.enable_simready_search:
        simready_config = simready_search_config_from_cli(
            sources=benchmark_case.simready_sources,
            s3_url=benchmark_case.simready_s3_url,
            service_url=benchmark_case.simready_service_url,
            max_results_per_object=benchmark_case.simready_max_results_per_object,
        )
    agent = EnvironmentGenerationAgent(
        endpoint=endpoint.name,
        model=args.model,
        temperature=args.temperature,
        enable_simready_search=benchmark_case.enable_simready_search if benchmark_case is not None else False,
        simready_config=simready_config,
    )
    model = args.model or endpoint.model

    print("[benchmark] building catalogues outside the measured region", flush=True)
    asset_catalog = build_asset_catalogue()
    relation_catalog = build_relation_catalogue()
    task_catalog = build_task_catalogue()

    results: list[SampleResult] = []
    spec_output_dir = args.spec_output_dir
    for run_index in range(1, args.num_runs + 1):
        print(f"[benchmark] sample {run_index}/{args.num_runs}", flush=True)
        result = _run_sample(
            agent,
            prompt,
            run_index,
            asset_catalog=asset_catalog,
            relation_catalog=relation_catalog,
            task_catalog=task_catalog,
            object_references_expected=(
                benchmark_case.object_references_expected if benchmark_case is not None else None
            ),
            spec_output_dir=spec_output_dir,
        )
        results.append(result)
        if result.generated_spec_path is not None:
            spec_output_dir = None
        if result.time_to_first_spec_ms is None:
            print(f"[benchmark] sample {run_index}: FAILED ({result.error})", flush=True)
        else:
            print(f"[benchmark] sample {run_index}: {result.time_to_first_spec_ms:.3f} ms", flush=True)

    summary = summarize_latencies([result.time_to_first_spec_ms for result in results])
    payload = {
        "type": "timing_benchmark",
        "name": "time_to_first_environment_spec",
        "definition": "time(env_spec_available) - time(request_sent)",
        "unit": "ms",
        "endpoint": endpoint.name,
        "model": model,
        "temperature": args.temperature,
        "case": benchmark_case.name if benchmark_case is not None else "custom",
        "cases_file": str(args.cases_file) if benchmark_case is not None else None,
        "simready": (
            {
                "enabled": benchmark_case.enable_simready_search,
                "sources": benchmark_case.simready_sources,
                "s3_url": benchmark_case.simready_s3_url,
                "service_url": benchmark_case.simready_service_url,
                "max_results_per_object": benchmark_case.simready_max_results_per_object,
            }
            if benchmark_case is not None
            else None
        ),
        "object_references_expected": benchmark_case.object_references_expected if benchmark_case is not None else None,
        "prompt": prompt,
        "spec_output_dir": str(args.spec_output_dir) if args.spec_output_dir is not None else None,
        "summary": summary,
        "samples": [asdict(result) for result in results],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\nTime to first environment spec", flush=True)
    print(f"  successful: {summary['successful_samples']}/{summary['requested_samples']}", flush=True)
    for name in ("p50_ms", "p95_ms", "p99_ms"):
        value = summary[name]
        formatted = "n/a" if value is None else f"{value:.3f} ms"
        print(f"  {name.removesuffix('_ms')}: {formatted}", flush=True)
    print(f"  results: {args.output_path}", flush=True)
    return 1 if summary["failed_samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
