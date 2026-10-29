# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success across the RoboLab and KitchenBench environment specs.

Two rates are reported: the share of valid candidates in each build's first placement pool batch, which pass every
required pre-physics check, and the share of environments whose pool holds only valid layouts, without a best-loss
fallback.

Run inside the container from the repository root, and write a results file:

    python -m isaaclab_arena_benchmarking.object_placement.run_placement_success_benchmark \\
        --output outputs/placement_success/run.json
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import isaaclab_arena
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_benchmarking.object_placement.measurement import (
    REPO_ROOT,
    SUITE_SPEC_GLOBS,
    discover_specs,
    get_git_provenance,
    get_seed_block_index,
    measure_environment,
)
from isaaclab_arena_benchmarking.object_placement.results import (
    EnvironmentStatus,
    PlacementBenchmarkResults,
    PlacementBenchmarkSettings,
    format_results,
)


def get_benchmark_parser() -> argparse.ArgumentParser:
    """Return a parser with the simulator launch flags and the benchmark's own flags."""
    defaults = PlacementBenchmarkSettings()
    parser = argparse.ArgumentParser(description="Measure relation-placement success across environment specs.")
    AppLauncher.add_app_launcher_args(parser)
    group = parser.add_argument_group("Placement Success Benchmark Arguments")
    group.add_argument(
        "--suite", choices=[*SUITE_SPEC_GLOBS, "all"], default="all", help="Environment suite to benchmark."
    )
    group.add_argument(
        "--env",
        action="append",
        default=None,
        help="Glob on spec file stems to select environments; repeat to select several.",
    )
    group.add_argument(
        "--num_envs", type=int, default=defaults.num_envs, help="Environments in each build, one build per seed."
    )
    group.add_argument(
        "--placement_seeds",
        type=int,
        nargs="+",
        default=defaults.placement_seeds,
        help="Distinct nonnegative seeds; each seed builds every spec once, solving its own candidates.",
    )
    group.add_argument(
        "--random_yaw_init",
        action="store_true",
        help="Sample a random initial yaw for placed objects, including a relation-placed robot.",
    )
    group.add_argument(
        "--collision_mode",
        choices=[mode.value for mode in CollisionMode],
        default=None,
        help="Default collision mode for placement, overriding the spec's placer parameters.",
    )
    group.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Results file to write, creating its folder; it is rewritten after each environment.",
    )
    return parser


def main() -> None:
    """Measure the selected specs and print the suite results, also writing them to --output when it is given."""
    args_cli = get_benchmark_parser().parse_args()
    settings = PlacementBenchmarkSettings(
        num_envs=args_cli.num_envs,
        placement_seeds=args_cli.placement_seeds,
        random_yaw_init=args_cli.random_yaw_init,
        collision_mode=None if args_cli.collision_mode is None else CollisionMode(args_cli.collision_mode),
    )
    specs = discover_specs(args_cli.suite, args_cli.env)
    assert specs, f"No environment specs match suite '{args_cli.suite}' and filters {args_cli.env}"
    # The specs and the recorded commit come from this checkout, so the measured code must too.
    package_root = Path(isaaclab_arena.__file__).resolve().parents[1]
    assert package_root == REPO_ROOT, f"isaaclab_arena is imported from {package_root}, not {REPO_ROOT}; set PYTHONPATH"
    commit, branch, dirty = get_git_provenance()
    if commit is None:
        print(f"[placement benchmark] Warning: git cannot read {REPO_ROOT}, so the results record no commit.")
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    results = PlacementBenchmarkResults(start_time, commit, branch, dirty, settings, [])

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena_environments.cli import ensure_environments_registered

        # The relation solver places on CUDA whenever it is available.
        results.settings.solver_device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        ensure_environments_registered()
        for index, (suite, spec_path) in enumerate(specs, start=1):
            print(f"[placement benchmark] {index}/{len(specs)} {suite}/{spec_path.stem}")
            seed_block_index = get_seed_block_index(suite, spec_path)
            results.environments.append(measure_environment(suite, spec_path, settings, seed_block_index))
            # Rewrite the results file after each environment so that a stopped run keeps its results.
            if args_cli.output is not None:
                results.save(args_cli.output)
        print(format_results(results))
        # Raising inside SimulationAppContext exits non-zero, so a run with failed environments reports failure.
        failed = [record.environment for record in results.environments if record.status == EnvironmentStatus.FAILED]
        if failed:
            raise SystemExit(f"{len(failed)} environment(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
