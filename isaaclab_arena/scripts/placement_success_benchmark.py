# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success across the RoboLab and KitchenBench environment specs.

Each spec is built once per placement seed with the normal ArenaEnvBuilder.build_env_cfg(), without stepping physics,
and the build's placement diagnostics are recorded. Two rates are reported: the share of valid candidates in the
placement pool's first batch, which pass every required pre-physics check, and the share of environments whose pool
holds only valid layouts, without a best-loss fallback.

Run inside the container and write a results file:

    python isaaclab_arena/scripts/placement_success_benchmark.py --output outputs/placement_success/before.json

Compare two runs, for example before and after a placement change, without starting the simulator:

    python isaaclab_arena/scripts/placement_success_benchmark.py \\
        --compare outputs/placement_success/before.json outputs/placement_success/after.json
"""

from __future__ import annotations

import argparse
import copy
import datetime
import fnmatch
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

import isaaclab_arena
from isaaclab_arena.offline_placement.placement_success import (
    EnvironmentRecord,
    EnvironmentStatus,
    PlacementBenchmarkResults,
    PlacementBenchmarkSettings,
    compare_results,
    format_comparison,
    format_provenance,
    format_results,
)
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_diagnostics import PlacementDiagnostics

_REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_SPEC_GLOBS = {
    "robolab": "isaaclab_arena_environments/robolab/tasks/*.yaml",
    "kitchen_bench": "isaaclab_arena_environments/kitchen_bench/*.yaml",
}
"""Suite name mapped to a glob over its environment graph-spec YAML files, relative to the repository root.

Append new suites, so that the specs of earlier suites keep their seed blocks.
"""

_BENCHMARKED_PACKAGES = ("isaaclab_arena", "isaaclab_arena_environments")
"""Packages whose code and specs the benchmark measures, relative to the repository root."""

_STAMPED_COMMIT_ENV_VAR = "ARENA_GIT_SHA"
"""Environment variable an Arena image can set to its commit, with a "-dirty" suffix for uncommitted changes."""

_STAMPED_BRANCH_ENV_VAR = "ARENA_GIT_BRANCH"
"""Environment variable an Arena image can set to its branch."""

_SEEDS_PER_ENVIRONMENT = 2**20
"""Size of each environment's block of candidate seeds, so environments never share candidates, even when they share
a scene."""


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
    group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        default=None,
        help="Compare two results files and exit without starting the simulator.",
    )
    return parser


def main() -> None:
    """Run the benchmark, or compare two results files with --compare."""
    args_cli = get_benchmark_parser().parse_args()

    if args_cli.compare is not None:
        before_path, after_path = args_cli.compare
        before = PlacementBenchmarkResults.load(before_path)
        after = PlacementBenchmarkResults.load(after_path)
        print(f"Before: {format_provenance(before)}")
        print(f"After: {format_provenance(after)}")
        print()
        print(format_comparison(compare_results(before, after)))
        return

    settings = PlacementBenchmarkSettings(
        num_envs=args_cli.num_envs,
        placement_seeds=args_cli.placement_seeds,
        random_yaw_init=args_cli.random_yaw_init,
        collision_mode=None if args_cli.collision_mode is None else CollisionMode(args_cli.collision_mode),
    )
    specs = discover_specs(args_cli.suite, args_cli.env)
    assert specs, f"No environment specs match suite '{args_cli.suite}' and filters {args_cli.env}"
    # The specs and the recorded commit come from this script's checkout, so the measured code must too.
    package_root = Path(isaaclab_arena.__file__).resolve().parents[1]
    assert (
        package_root == _REPO_ROOT
    ), f"isaaclab_arena is imported from {package_root}, not {_REPO_ROOT}; set PYTHONPATH"
    commit, branch, dirty = get_git_provenance()
    if commit is None:
        print(f"[placement benchmark] Warning: git cannot read {_REPO_ROOT}, so the results record no commit.")
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


def discover_specs(suite: str, env_filters: list[str] | None) -> list[tuple[str, Path]]:
    """Return (suite, spec path) pairs in a stable order, keeping specs that match any filter.

    Args:
        suite: Suite name, or "all" for every suite.
        env_filters: Globs on spec file stems, or None to keep every spec.
    """
    suites = list(SUITE_SPEC_GLOBS) if suite == "all" else [suite]
    specs = []
    for suite_name in suites:
        for spec_path in sorted(_REPO_ROOT.glob(SUITE_SPEC_GLOBS[suite_name])):
            if env_filters is None or any(fnmatch.fnmatch(spec_path.stem, pattern) for pattern in env_filters):
                specs.append((suite_name, spec_path))
    return specs


def get_git_provenance() -> tuple[str | None, str | None, bool | None]:
    """Return the checkout's commit, branch and whether it has uncommitted changes, or None where they are unknown.

    Changes count when they are to tracked files or are new files in the packages the benchmark reads, such as a new
    spec. Other untracked files, such as local tool settings that only a user's own git config ignores, do not. When
    git cannot read the checkout, as in an image without .git, the values come from the ARENA_GIT_SHA and
    ARENA_GIT_BRANCH environment variables if they are set.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        stamped_commit = os.environ.get(_STAMPED_COMMIT_ENV_VAR) or None
        if stamped_commit is None:
            return None, None, None
        stamped_branch = os.environ.get(_STAMPED_BRANCH_ENV_VAR) or None
        return stamped_commit.removesuffix("-dirty"), stamped_branch, stamped_commit.endswith("-dirty")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    tracked_changes = _run_git("status", "--porcelain", "--untracked-files=no")
    package_changes = _run_git("status", "--porcelain", "--", *_BENCHMARKED_PACKAGES)
    if tracked_changes is None or package_changes is None:
        return commit, branch, None
    return commit, branch, bool(tracked_changes or package_changes)


def get_seed_block_index(suite: str, spec_path: Path) -> int:
    """Return the index of a spec's block of candidate seeds, numbered over every spec of every suite.

    Numbering over every spec gives a filtered run the same candidates as a full run.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML, as returned by discover_specs().
    """
    all_specs = discover_specs("all", None)
    # CPU torch generators use only the low 32 bits of a seed, so every block must end below 2**32.
    assert len(all_specs) * _SEEDS_PER_ENVIRONMENT <= 2**32, "Too many specs for separate seed blocks"
    return all_specs.index((suite, spec_path))


def measure_environment(
    suite: str, spec_path: Path, settings: PlacementBenchmarkSettings, seed_block_index: int
) -> EnvironmentRecord:
    """Return one environment spec's record after building it once per seed, recording failures instead of raising.

    Each build is the normal ArenaEnvBuilder.build_env_cfg() for settings.num_envs environments, without stepping
    physics; the record accumulates the placement diagnostics of every build.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML.
        settings: Benchmark settings.
        seed_block_index: Index of the environment's own block of candidate seeds, from get_seed_block_index().
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    start = time.perf_counter()
    record = EnvironmentRecord(suite=suite, environment=spec_path.stem, status=EnvironmentStatus.SCORED)
    try:
        for seed in settings.placement_seeds:
            # A build applies build-time variations to its environment, so every build starts from the spec.
            arena_env = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env()
            # Graph specs always carry placer parameters, built from the defaults and the spec's overrides.
            assert arena_env.placer_params is not None, f"{spec_path} has no placer parameters"
            if settings.random_yaw_init or settings.collision_mode is not None:
                arena_env.placer_params = get_overridden_placer_params(arena_env.placer_params, settings)
            placement_seed = get_placement_seed(arena_env.placer_params, settings.num_envs, seed_block_index, seed)
            builder_cfg = ArenaEnvBuilderCfg(num_envs=settings.num_envs, placement_seed=placement_seed)
            builder = ArenaEnvBuilder(arena_env, builder_cfg)
            builder.build_env_cfg()
            if builder.placement_diagnostics is None:
                record.status = EnvironmentStatus.SKIPPED
                record.reason = "solves no relations"
                break
            add_build_results(record, builder.placement_diagnostics, settings.num_envs)
    except Exception as exc:
        traceback.print_exc()
        record.status = EnvironmentStatus.FAILED
        record.reason = f"{type(exc).__name__}: {exc}"
    record.wall_time_s = time.perf_counter() - start
    return record


def get_overridden_placer_params(
    placer_params: ObjectPlacerParams, settings: PlacementBenchmarkSettings
) -> ObjectPlacerParams:
    """Return a copy of an environment's placer parameters with the benchmark's yaw and collision-mode overrides.

    Args:
        placer_params: The environment's placer parameters.
        settings: Benchmark settings.
    """
    params = copy.copy(placer_params)
    params.solver_params = copy.copy(params.solver_params)
    if settings.random_yaw_init:
        params.random_yaw_init = True
    if settings.collision_mode is not None:
        params.solver_params.collision_mode = settings.collision_mode
    return params


def get_placement_seed(placer_params: ObjectPlacerParams, num_envs: int, seed_block_index: int, seed: int) -> int:
    """Return one build's placement seed, so that every build of every spec solves its own candidates.

    A placement pool solves at most max_placement_attempts batches, each of at most
    num_envs * min_unique_layouts_per_env * max_placement_attempts candidates, seeded consecutively from the
    placement seed.

    Args:
        placer_params: The placer parameters the build uses.
        num_envs: Environments per build.
        seed_block_index: Index of the environment's own block of candidate seeds.
        seed: Nonnegative benchmark seed.
    """
    attempts = placer_params.max_placement_attempts
    seeds_per_build = attempts * num_envs * placer_params.min_unique_layouts_per_env * attempts
    assert (seed + 1) * seeds_per_build <= _SEEDS_PER_ENVIRONMENT, (
        f"Seed {seed} needs {(seed + 1) * seeds_per_build} candidate seeds, more than the {_SEEDS_PER_ENVIRONMENT}"
        " reserved per environment; use smaller seeds or fewer environments"
    )
    return seed_block_index * _SEEDS_PER_ENVIRONMENT + seed * seeds_per_build


def add_build_results(record: EnvironmentRecord, diagnostics: PlacementDiagnostics, num_envs: int) -> None:
    """Accumulate one build's placement diagnostics into the environment record.

    Candidate counts come from the pool's first batch, which has the same size in every build; later batches only
    refill environments that lacked valid layouts, so including them would favor scenes that need more batches.

    Args:
        record: Record to update in place.
        diagnostics: Placement diagnostics of the build.
        num_envs: Environments in the build.
    """
    first_batch = diagnostics.batches[0]
    record.num_builds += 1
    record.num_envs += num_envs
    record.num_batches += len(diagnostics.batches)
    record.num_fallback_envs += len(set(diagnostics.fallback_envs))
    record.num_candidates += first_batch.num_candidates
    record.num_valid_candidates += first_batch.num_valid
    for check, counts in first_batch.checks.items():
        record.rejections_per_check[check] = record.rejections_per_check.get(check, 0) + counts.num_rejected
    build_checks = {check: counts.required for check, counts in first_batch.checks.items()}
    assert not record.checks or record.checks == build_checks, f"Checks differ between builds of {record.environment}"
    record.checks = build_checks


def _run_git(*args: str) -> str | None:
    """Return git's stripped output in the repository, or None when git is unavailable or fails."""
    # Trust the checkout even when it belongs to another user, as it does for root inside the container.
    command = ["git", "-c", f"safe.directory={_REPO_ROOT}", *args]
    try:
        completed = subprocess.run(command, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    main()
