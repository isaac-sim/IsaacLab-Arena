# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success across the RoboLab and KitchenBench environment specs.

Each spec's placement inputs are built as ArenaEnvBuilder builds them, without stepping physics, then placed for every
placement seed. Two rates are reported: the share of valid candidates, which pass every required pre-physics check
and are what an environment build's placement pool keeps, and the share of layouts whose best of
max_placement_attempts candidates is valid, as in ObjectPlacer.place().

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
    from isaaclab_arena.environments.arena_env_builder import LivePlacementInputs
    from isaaclab_arena.relations.placement_result import PlacementResult

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
        "--num_layouts", type=int, default=defaults.num_layouts, help="Layouts placed per environment and seed."
    )
    group.add_argument(
        "--placement_seeds",
        type=int,
        nargs="+",
        default=defaults.placement_seeds,
        help="Distinct nonnegative placement seeds; each seed solves its own disjoint set of candidates.",
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
        num_layouts=args_cli.num_layouts,
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
    """Return the checkout's commit, branch and whether it has uncommitted changes, or None where git cannot tell.

    Changes count when they are to tracked files or are new files in the packages the benchmark reads, such as a new
    spec. Other untracked files, such as local tool settings that only a user's own git config ignores, do not.
    """
    commit = _run_git("rev-parse", "HEAD")
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
    """Return one environment spec's record after placing every seed, recording failures instead of raising.

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
        arena_env = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env()
        inputs = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg()).prepare_live_placement_inputs()
        if all(asset.is_anchor for asset in inputs.assets):
            record.status = EnvironmentStatus.SKIPPED
            record.reason = "places nothing other than anchors"
        else:
            placer = CandidatePlacer(inputs, settings, seed_block_index)
            for seed in settings.placement_seeds:
                add_seed_results(record, placer.place(seed))
    except Exception as exc:
        traceback.print_exc()
        record.status = EnvironmentStatus.FAILED
        record.reason = f"{type(exc).__name__}: {exc}"
    record.wall_time_s = time.perf_counter() - start
    return record


class CandidatePlacer:
    """Places one environment's assets for each benchmark seed, returning every candidate."""

    def __init__(self, inputs: LivePlacementInputs, settings: PlacementBenchmarkSettings, seed_block_index: int):
        """Copy the placer parameters, apply the benchmark settings, build the placer and find the fixed obstacles.

        Object sets among the assets get their members for every layout here, from the start of the seed block.

        Args:
            inputs: The environment's placement inputs; its placer parameters are copied, not modified.
            settings: Benchmark settings.
            seed_block_index: Index of the environment's own block of candidate seeds.
        """
        from isaaclab_arena.relations.geometry.bounding_box_helpers import assign_variants_for_envs
        from isaaclab_arena.relations.geometry.passive_collision_objects import get_placement_collision_objects
        from isaaclab_arena.relations.object_placer import ObjectPlacer

        params = copy.copy(inputs.placer_params)
        params.solver_params = copy.copy(params.solver_params)
        params.reachability_config = copy.copy(params.reachability_config)
        if settings.random_yaw_init:
            params.random_yaw_init = True
        if settings.collision_mode is not None:
            params.solver_params.collision_mode = settings.collision_mode
        # A layout normally keeps the best of max_placement_attempts candidates. Solve one candidate per attempt and
        # request all of them, so the same candidates come back ranked and the valid ones can be counted.
        self._attempts = params.max_placement_attempts
        params.max_placement_attempts = 1
        # Validators are built once here; place() only changes the placer's seed.
        self._placer = ObjectPlacer(params)
        self._assets = inputs.assets
        self._collision_objects = get_placement_collision_objects(
            inputs.assets, inputs.scene_assets, params.solver_params.collision_mode
        )
        self._num_layouts = settings.num_layouts
        self._first_seed = seed_block_index * _SEEDS_PER_ENVIRONMENT
        # An object set keeps its first member assignment, so draw it once from this environment's seed block; otherwise
        # every seed would keep the members of whichever seed ran first.
        assign_variants_for_envs(self._assets, self._num_layouts, placement_seed=self._first_seed)

    def place(self, seed: int) -> list[list[PlacementResult]]:
        """Place num_layouts layouts for one seed and return every candidate, ranked per layout.

        The layouts match ObjectPlacer.place() with placement seed
        seed_block_index * 2**20 + seed * num_layouts * max_placement_attempts, except that an object set keeps the
        members drawn from the start of the seed block for every seed. Candidates are seeded consecutively from the
        placement seed, so different seeds and different seed blocks solve disjoint candidates.

        Args:
            seed: Nonnegative benchmark seed.

        Returns:
            For each layout, its max_placement_attempts candidates, valid and lower-loss ones first.
        """
        candidates_per_seed = self._num_layouts * self._attempts
        assert (seed + 1) * candidates_per_seed <= _SEEDS_PER_ENVIRONMENT, (
            f"Seed {seed} needs {(seed + 1) * candidates_per_seed} candidate seeds, more than the"
            f" {_SEEDS_PER_ENVIRONMENT} reserved per environment; use smaller seeds or fewer layouts"
        )
        self._placer.params.placement_seed = self._first_seed + seed * candidates_per_seed
        return self._placer.place_ranked_per_env(
            self._assets,
            num_envs=self._num_layouts,
            results_per_env=self._attempts,
            collision_objects=self._collision_objects,
        )


def add_seed_results(record: EnvironmentRecord, ranked_per_layout: list[list[PlacementResult]]) -> None:
    """Accumulate one seed's layouts and candidates into the environment record.

    Args:
        record: Record to update in place.
        ranked_per_layout: For each layout, its candidates with the best one first.
    """
    for ranked in ranked_per_layout:
        best = ranked[0]
        record.num_layouts += 1
        record.num_candidates += len(ranked)
        record.num_valid_candidates += sum(result.success for result in ranked)
        for result in ranked:
            for check in result.validation_results.get_failed_validation_check_names:
                record.candidate_failures_per_check[check] = record.candidate_failures_per_check.get(check, 0) + 1
        validation = best.validation_results
        if best.success:
            record.num_successful_layouts += 1
        layout_checks = {}
        for check in validation.validation_results:
            # As in PlacementValidationResults, no declared required checks means every check is required.
            layout_checks[str(check)] = validation.required_checks is None or check in validation.required_checks
        assert (
            not record.checks or record.checks == layout_checks
        ), f"Checks differ between layouts of {record.environment}"
        record.checks = layout_checks


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
