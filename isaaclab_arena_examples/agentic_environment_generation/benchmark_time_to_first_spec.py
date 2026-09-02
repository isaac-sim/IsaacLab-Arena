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
        isaaclab_arena_examples/agentic_environment_generation/benchmark_time_to_first_spec.py \
        --inference_endpoint internal --num_runs 100 \
        --output_path outputs/time_to_first_spec.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from isaaclab_arena.agentic_environment_generation.catalogues import (
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.environment_generation_agent import EnvironmentGenerationAgent
from isaaclab_arena.agentic_environment_generation.inference_backend import resolve_inference_endpoint
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadySourceKind,
    simready_search_config_from_cli,
)
from isaaclab_arena.agentic_environment_generation.spec_io import write_env_graph_spec
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
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
    simready_source: str = SimReadySourceKind.ISAAC_SIM_GA.value
    simready_s3_url: str | None = None
    simready_service_url: str | None = None
    simready_max_results_per_object: int = 1


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

    cases: dict[str, BenchmarkCase] = {}
    for name, values in raw_cases.items():
        assert isinstance(name, str) and name, f"benchmark case names must be non-empty strings: {name!r}"
        assert isinstance(values, dict), f"benchmark case {name!r} must contain a mapping"
        known_fields = {
            "prompt",
            "object_references_expected",
            "enable_simready_search",
            "simready_source",
            "simready_s3_url",
            "simready_service_url",
            "simready_max_results_per_object",
        }
        unexpected = set(values) - known_fields
        assert not unexpected, f"benchmark case {name!r} has unknown fields: {sorted(unexpected)}"
        prompt = values.get("prompt")
        assert isinstance(prompt, str) and prompt.strip(), f"benchmark case {name!r} must define a non-empty prompt"
        object_references_expected = values.get("object_references_expected")
        assert object_references_expected is None or isinstance(
            object_references_expected, bool
        ), f"benchmark case {name!r} object_references_expected must be true, false, or omitted"
        enable_simready_search = values.get("enable_simready_search", False)
        assert isinstance(
            enable_simready_search, bool
        ), f"benchmark case {name!r} enable_simready_search must be true or false"
        simready_source = values.get("simready_source", SimReadySourceKind.ISAAC_SIM_GA.value)
        assert simready_source in tuple(
            kind.value for kind in SimReadySourceKind
        ), f"benchmark case {name!r} simready_source must be one of {[kind.value for kind in SimReadySourceKind]}"
        simready_s3_url = values.get("simready_s3_url")
        assert simready_s3_url is None or isinstance(
            simready_s3_url, str
        ), f"benchmark case {name!r} simready_s3_url must be a string or omitted"
        simready_service_url = values.get("simready_service_url")
        assert simready_service_url is None or isinstance(
            simready_service_url, str
        ), f"benchmark case {name!r} simready_service_url must be a string or omitted"
        simready_max_results = values.get("simready_max_results_per_object", 1)
        assert (
            isinstance(simready_max_results, int) and simready_max_results > 0
        ), f"benchmark case {name!r} simready_max_results_per_object must be a positive integer"
        cases[name] = BenchmarkCase(
            name=name,
            prompt=prompt.strip(),
            object_references_expected=object_references_expected,
            enable_simready_search=enable_simready_search,
            simready_source=simready_source,
            simready_s3_url=simready_s3_url,
            simready_service_url=simready_service_url,
            simready_max_results_per_object=simready_max_results,
        )

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
    final_spec_accepted: bool
    error: str | None = None
    generated_spec_path: str | None = None


def percentile(values: list[float], p: float) -> float | None:
    """Return the nearest-rank percentile, or None when ``values`` is empty.

    Args:
        values: Measurements to summarize.
        p: Percentile in the range ``[0, 100]``.

    Returns:
        The nearest-rank percentile value.
    """
    assert 0 <= p <= 100, f"percentile must be in [0, 100], got {p}"
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


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
    if spec is not None and spec_output_dir is not None:
        generated_spec_path = str(write_env_graph_spec(spec, spec_output_dir))
    return SampleResult(
        run_index=run_index,
        time_to_first_spec_ms=latency_ms,
        final_spec_accepted=spec is not None,
        error=error,
        generated_spec_path=generated_spec_path,
    )


def _summary(results: list[SampleResult]) -> dict[str, int | float | None]:
    latencies = [result.time_to_first_spec_ms for result in results if result.time_to_first_spec_ms is not None]
    return {
        "requested_samples": len(results),
        "successful_samples": len(latencies),
        "failed_samples": len(results) - len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


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
        help="Optional directory for the first generated spec, named from its env_name.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and write sample-level results plus p50/p95/p99."""
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
            source=benchmark_case.simready_source,
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
            spec_output_dir=args.spec_output_dir if run_index == 1 else None,
        )
        results.append(result)
        if result.time_to_first_spec_ms is None:
            print(f"[benchmark] sample {run_index}: FAILED ({result.error})", flush=True)
        else:
            print(f"[benchmark] sample {run_index}: {result.time_to_first_spec_ms:.3f} ms", flush=True)

    summary = _summary(results)
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
                "source": benchmark_case.simready_source,
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
    return 0 if summary["successful_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
