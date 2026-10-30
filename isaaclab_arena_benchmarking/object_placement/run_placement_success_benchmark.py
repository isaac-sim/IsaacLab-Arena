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
    measure_spec,
)
from isaaclab_arena_benchmarking.object_placement.results import (
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
    group.add_argument("--suite", choices=[*SUITE_SPEC_GLOBS, "all"], default="all", help="Spec suite to benchmark.")
    group.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Glob on spec file stems to select specs; repeat to select several.",
    )
    group.add_argument(
        "--num_envs", type=int, default=defaults.num_envs, help="Environments in each build, one build per seed index."
    )
    group.add_argument(
        "--seed_indices",
        type=int,
        nargs="+",
        default=defaults.seed_indices,
        help="Distinct nonnegative indices; each one builds every spec once, from its own range of candidate seeds.",
    )
    group.add_argument(
        "--random_yaw_init",
        action="store_true",
        default=defaults.random_yaw_init,
        help=(
            "Sample a random initial yaw for placed objects, including a relation-placed robot; without it, each"
            " spec's own setting applies."
        ),
    )
    group.add_argument(
        "--collision_mode",
        choices=[mode.value for mode in CollisionMode],
        default=None,
        help="Solver's default collision mode, replacing the spec's; objects that set their own mode keep it.",
    )
    group.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Results file to write, creating its folder; it is rewritten after each spec.",
    )
    return parser


def main() -> None:
    """Measure the selected specs and print the suite results, also writing them to --output when it is given."""
    args_cli = get_benchmark_parser().parse_args()
    settings = PlacementBenchmarkSettings(
        num_envs=args_cli.num_envs,
        seed_indices=args_cli.seed_indices,
        random_yaw_init=args_cli.random_yaw_init,
        collision_mode=None if args_cli.collision_mode is None else CollisionMode(args_cli.collision_mode),
    )
    specs = discover_specs(args_cli.suite, args_cli.spec)
    assert specs, f"No specs match suite '{args_cli.suite}' and filters {args_cli.spec}"
    # The specs and the recorded commit come from this checkout, so the measured code must too.
    package_root = Path(isaaclab_arena.__file__).resolve().parents[1]
    assert package_root == REPO_ROOT, f"isaaclab_arena is imported from {package_root}, not {REPO_ROOT}; set PYTHONPATH"
    commit, branch, dirty = get_git_provenance()
    if commit is None:
        print(f"[placement benchmark] Warning: git cannot read {REPO_ROOT}, so the results record no commit.")
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena_environments.cli import ensure_environments_registered

        # The relation solver places on CUDA whenever it is available.
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        selected_specs = [spec_path.stem for _, spec_path in specs]
        results = PlacementBenchmarkResults(
            start_time=start_time,
            commit=commit,
            branch=branch,
            dirty=dirty,
            gpu_name=gpu_name,
            settings=settings,
            selected_specs=selected_specs,
            records=[],
        )
        ensure_environments_registered()
        for index, (suite, spec_path) in enumerate(specs, start=1):
            print(f"[placement benchmark] {index}/{len(specs)} {suite}/{spec_path.stem}")
            results.records.append(measure_spec(suite, spec_path, settings))
            # Rewrite the results file after each spec so that a stopped run keeps its results.
            if args_cli.output is not None:
                results.save(args_cli.output)
        print(format_results(results))
        # Raising inside SimulationAppContext exits non-zero, so a run with failed specs reports failure.
        failed = [record.spec for record in results.records if record.error is not None]
        if failed:
            raise SystemExit(f"{len(failed)} spec(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
