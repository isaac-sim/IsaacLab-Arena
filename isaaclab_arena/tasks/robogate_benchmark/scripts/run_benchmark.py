#!/usr/bin/env python3
"""Run the RoboGate 68-scenario benchmark on Isaac Lab-Arena.

Usage:
    # Full benchmark with Isaac Lab-Arena
    python scripts/run_benchmark.py --embodiment franka --config configs/robogate_68.yaml

    # Mock mode (no simulator needed)
    python scripts/run_benchmark.py --mock --output results/mock_results.json

    # With Isaac Lab-Arena environment builder
    python scripts/run_benchmark.py --embodiment franka --enable-cameras

Exit codes:
    0: PASS (confidence >= 76)
    1: FAIL (confidence < 76)
    2: ERROR
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robogate_benchmark.confidence_scorer import compute_confidence_score
from robogate_benchmark.environments import RoboGateBenchmarkEnvironment
from robogate_benchmark.metrics import (
    CycleResult,
    collect_failure_evidence,
    compute_all_metrics,
    compute_scenario_summary,
    evaluate_all_metrics,
)
from robogate_benchmark.report_generator import (
    generate_arena_metrics,
    generate_json_report,
    generate_text_report,
)
from robogate_benchmark.scenarios import (
    VARIANT_CONFIGS,
    build_scenario_suite,
)

logger = logging.getLogger(__name__)


def run_mock_benchmark(
    seed: int = 42,
    nominal_count: int = 20,
    edge_count: int = 15,
    adversarial_count: int = 10,
) -> list[CycleResult]:
    """Run mock benchmark with synthetic scripted-policy results.

    Simulates the scripted controller performance profile:
        nominal: 95-100% SR
        edge_cases: 70-85% SR
        adversarial: 40-60% SR
        domain_randomization: 85-95% SR

    Args:
        seed: Random seed.
        nominal_count: Number of nominal scenarios.
        edge_count: Number of edge case scenarios.
        adversarial_count: Number of adversarial scenarios.

    Returns:
        List of CycleResult for all 68 scenarios.
    """
    rng = np.random.default_rng(seed)
    scenarios = build_scenario_suite(
        seed=seed,
        nominal_count=nominal_count,
        edge_count=edge_count,
        adversarial_count=adversarial_count,
    )

    # Success rate profiles for scripted controller
    sr_profiles = {
        "nominal": {
            "standard_objects": 0.98,
            "standard_lighting": 0.97,
            "centered_placement": 1.00,
        },
        "edge_cases": {
            "small_objects": 0.70,
            "heavy_objects": 0.75,
            "edge_placement": 0.80,
            "occluded_objects": 0.73,
            "transparent_objects": 0.72,
        },
        "adversarial": {
            "low_lighting": 0.50,
            "cluttered_scene": 0.55,
            "slippery_surface": 0.45,
            "moving_disturbance": 0.40,
        },
        "domain_randomization": {
            "lighting": 0.90,
            "object_color": 0.93,
            "position": 0.95,
            "camera": 0.88,
        },
    }

    cycles: list[CycleResult] = []

    for scenario in scenarios:
        cat = scenario.category.value
        var = scenario.variant
        profile = sr_profiles.get(cat, {})
        p_success = profile.get(var, 0.80)

        success = rng.random() < p_success
        collision = not success and rng.random() < 0.15
        drop = not success and not collision and rng.random() < 0.20
        grasp_miss = not success and not collision and not drop

        cycle_time = float(rng.uniform(5.0, 10.0)) if success else float(rng.uniform(2.0, 15.0))

        cycles.append(
            CycleResult(
                scenario_category=cat,
                scenario_variant=var,
                success=success,
                cycle_time=round(cycle_time, 3),
                collision=collision,
                drop=drop,
                grasp_miss=grasp_miss,
            )
        )

    return cycles


def run_arena_benchmark(args: argparse.Namespace) -> list[CycleResult]:
    """Run benchmark using Isaac Lab-Arena environment.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of CycleResult from all scenarios.
    """
    env_def = RoboGateBenchmarkEnvironment()
    arena_env = env_def.get_env(args)

    try:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.cli import get_isaaclab_arena_cli_parser

        builder = ArenaEnvBuilder(arena_env, args)
        env = builder.make_registered()
    except ImportError:
        logger.error(
            "Isaac Lab-Arena not available. Use --mock for testing."
        )
        sys.exit(2)

    scenarios = build_scenario_suite(
        seed=args.seed,
        nominal_count=args.nominal_count,
        edge_count=args.edge_count,
        adversarial_count=args.adversarial_count,
    )

    cycles: list[CycleResult] = []

    for i, scenario in enumerate(scenarios):
        obs, info = env.reset()

        done = False
        steps = 0
        max_steps = 300
        success = False

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if info.get("success", False):
                success = True
                break

        collision = info.get("collision", False)
        drop = info.get("drop", False)
        grasp_miss = not success and not collision and not drop
        cycle_time = info.get("cycle_time", steps * 0.05)

        cycles.append(
            CycleResult(
                scenario_category=scenario.category.value,
                scenario_variant=scenario.variant,
                success=success,
                cycle_time=round(cycle_time, 3),
                collision=collision,
                drop=drop,
                grasp_miss=grasp_miss,
            )
        )

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(scenarios))

    env.close()
    return cycles


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RoboGate 68-Scenario Benchmark for Isaac Lab-Arena"
    )
    RoboGateBenchmarkEnvironment.add_cli_args(parser)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock benchmark (no simulator)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robogate_68.yaml",
        help="Benchmark config path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/<timestamp>.json)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline results JSON for diff comparison",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress text report output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    t0 = time.time()

    # Run benchmark
    if args.mock:
        logger.info("Running mock benchmark (no simulator)")
        cycles = run_mock_benchmark(
            seed=args.seed,
            nominal_count=args.nominal_count,
            edge_count=args.edge_count,
            adversarial_count=args.adversarial_count,
        )
    else:
        logger.info("Running Isaac Lab-Arena benchmark")
        cycles = run_arena_benchmark(args)

    total_time = time.time() - t0

    # Compute metrics
    raw_metrics = compute_all_metrics(cycles)

    # Load baseline if provided
    baseline_metrics = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})

    evaluated = evaluate_all_metrics(raw_metrics, baseline_metrics)
    summaries = compute_scenario_summary(cycles)
    confidence = compute_confidence_score(evaluated, summaries, baseline_metrics)
    evidence = collect_failure_evidence(cycles)

    passed = sum(1 for c in cycles if c.success)
    total = len(cycles)

    # Build results
    results = {
        "metadata": {
            "benchmark": "robogate",
            "version": "1.0.0",
            "mode": "mock" if args.mock else "arena",
            "embodiment": args.embodiment,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_scenarios": total,
            "total_time_s": round(total_time, 1),
            "config": args.config,
        },
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": round(passed / total, 4) if total else 0,
            "confidence_score": confidence["score"],
            "verdict": confidence["verdict"],
        },
        "metrics": raw_metrics,
        "metrics_evaluated": {
            mid: {k: v for k, v in m.items()} for mid, m in evaluated.items()
        },
        "scenario_summary": {
            cat: {
                "total": s.total,
                "passed": s.passed,
                "failed": s.failed,
                "success_rate": round(s.pass_rate, 4),
            }
            for cat, s in summaries.items()
        },
        "confidence": confidence,
        "failure_evidence": [
            {
                "scenario": e.scenario,
                "failure_type": e.failure_type,
                "count": e.count,
                "severity": e.severity,
                "description": e.description,
            }
            for e in evidence
        ],
    }

    # Output
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/robogate_{ts}.json"

    output_path = generate_json_report(results, args.output)
    logger.info("Results: %s", output_path)

    if not args.quiet:
        print(generate_text_report(results))

    # Exit code
    verdict = confidence["verdict"]
    if verdict == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
