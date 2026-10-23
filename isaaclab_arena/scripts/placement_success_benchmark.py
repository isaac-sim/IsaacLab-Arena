# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success across the RoboLab and KitchenBench environment specs.

Each spec's placement inputs are built as ArenaEnvBuilder builds them, without stepping physics, then placed for every
placement seed. A layout succeeds when the best of its max_placement_attempts candidates passes every required
pre-physics check, as in ObjectPlacer.place(). The share of valid candidates is also reported, because it changes
before the best-of-candidates rate does.

Run inside the container and write a results file:

    /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py --suite all --output placement_main.json

Compare two runs, for example main against a placement change, without starting the simulator:

    /isaac-sim/python.sh isaaclab_arena/scripts/placement_success_benchmark.py \\
        --compare placement_main.json placement_change.json
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
    PlacementBenchmarkRun,
    compare_runs,
    format_comparison,
    format_provenance,
    format_run,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_mode import CollisionMode
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUITE_SPECS = {
    "robolab": "isaaclab_arena_environments/robolab/tasks/*.yaml",
    "kitchen_bench": "isaaclab_arena_environments/kitchen_bench/*.yaml",
}
_SEED_STREAM_LENGTH = 2**20
"""Candidate seeds reserved for each environment, so environments never share candidates, even when they share a
scene."""


def get_benchmark_parser() -> argparse.ArgumentParser:
    """Return a parser with the simulator launch flags and the benchmark's own flags."""
    parser = argparse.ArgumentParser(description="Measure relation-placement success across environment specs.")
    AppLauncher.add_app_launcher_args(parser)
    group = parser.add_argument_group("Placement Success Benchmark Arguments")
    group.add_argument("--suite", choices=[*_SUITE_SPECS, "all"], default="all", help="Environment suite to benchmark.")
    group.add_argument(
        "--env",
        action="append",
        default=None,
        help="Glob on spec file stems to select environments; repeat to select several.",
    )
    group.add_argument("--num_layouts", type=int, default=50, help="Layouts placed per environment and seed.")
    group.add_argument(
        "--placement_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Distinct nonnegative placement seeds; each seed solves its own disjoint set of candidates.",
    )
    group.add_argument(
        "--random_yaw_init",
        action="store_true",
        help="Sample a random initial yaw for placed objects, including a relation-placed robot.",
    )
    group.add_argument(
        "--collision_mode",
        choices=["bbox", "mesh"],
        default=None,
        help="Default collision mode for placement, overriding the spec's placer parameters.",
    )
    group.add_argument(
        "--output", type=Path, default=None, help="Results file to write; it is rewritten after each environment."
    )
    group.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CANDIDATE"),
        default=None,
        help="Compare two results files and exit without starting the simulator.",
    )
    return parser


def main() -> None:
    """Run the benchmark, or compare two results files with --compare."""
    args_cli = get_benchmark_parser().parse_args()

    if args_cli.compare is not None:
        baseline_path, candidate_path = args_cli.compare
        baseline, candidate = PlacementBenchmarkRun.load(baseline_path), PlacementBenchmarkRun.load(candidate_path)
        print(f"Baseline: {format_provenance(baseline)}")
        print(f"Candidate: {format_provenance(candidate)}")
        print()
        print(format_comparison(compare_runs(baseline, candidate)))
        return

    assert args_cli.num_layouts > 0, "--num_layouts must be positive"
    assert all(seed >= 0 for seed in args_cli.placement_seeds), "--placement_seeds must be nonnegative"
    # A repeated seed would count the same candidates twice.
    assert len(set(args_cli.placement_seeds)) == len(args_cli.placement_seeds), "--placement_seeds must be distinct"
    assert args_cli.output is None or args_cli.output.parent.is_dir(), f"No directory for --output {args_cli.output}"
    specs = discover_specs(args_cli.suite, args_cli.env)
    assert specs, f"No environment specs match suite '{args_cli.suite}' and filters {args_cli.env}"
    # The specs and the recorded commit come from this script's checkout, so the measured code must too.
    package_root = Path(isaaclab_arena.__file__).resolve().parents[1]
    assert (
        package_root == _REPO_ROOT
    ), f"isaaclab_arena is imported from {package_root}, not {_REPO_ROOT}; set PYTHONPATH"
    commit, branch, dirty = get_git_provenance(args_cli.output)
    if commit is None:
        print(f"[placement benchmark] Warning: git cannot read {_REPO_ROOT}, so the results record no commit.")
    settings = {
        "suite": args_cli.suite,
        "env_filters": args_cli.env,
        "num_layouts": args_cli.num_layouts,
        "placement_seeds": args_cli.placement_seeds,
        "random_yaw_init": args_cli.random_yaw_init,
        "collision_mode": args_cli.collision_mode,
    }
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    run = PlacementBenchmarkRun(start_time, commit, branch, dirty, settings, [])

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena_environments.cli import ensure_environments_registered

        # The relation solver places on CUDA whenever it is available.
        run.settings["solver_device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        ensure_environments_registered()
        for index, (suite, spec_path) in enumerate(specs, start=1):
            print(f"[placement benchmark] {index}/{len(specs)} {suite}/{spec_path.stem}")
            run.environments.append(measure_environment(suite, spec_path, args_cli))
            # Rewrite the results file after each environment so that a stopped run keeps its results.
            if args_cli.output is not None:
                run.save(args_cli.output)
        print(format_run(run))
        # Raising inside SimulationAppContext exits non-zero, so a run with failed environments reports failure.
        failed = [record.environment for record in run.environments if record.status == EnvironmentStatus.FAILED]
        if failed:
            raise SystemExit(f"{len(failed)} environment(s) failed: {', '.join(failed)}")


def discover_specs(suite: str, env_filters: list[str] | None) -> list[tuple[str, Path]]:
    """Return (suite, spec path) pairs in a stable order, keeping specs that match any filter.

    Args:
        suite: Suite name, or "all" for every suite.
        env_filters: Globs on spec file stems, or None to keep every spec.
    """
    suites = list(_SUITE_SPECS) if suite == "all" else [suite]
    specs = []
    for suite_name in suites:
        for spec_path in sorted(_REPO_ROOT.glob(_SUITE_SPECS[suite_name])):
            if env_filters is None or any(fnmatch.fnmatch(spec_path.stem, pattern) for pattern in env_filters):
                specs.append((suite_name, spec_path))
    return specs


def get_git_provenance(output: Path | None = None) -> tuple[str | None, str | None, bool | None]:
    """Return the checkout's commit, branch and dirty flag, or None for each value git cannot provide.

    Args:
        output: The benchmark's own output file, which does not count as a change when it is in the checkout.
    """
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status_args = ["status", "--porcelain"]
    if output is not None and output.resolve().is_relative_to(_REPO_ROOT):
        status_args += ["--", ".", f":(exclude,literal){output.resolve().relative_to(_REPO_ROOT)}"]
    status = _run_git(*status_args)
    return commit, branch, None if status is None else bool(status)


def get_seed_stream(suite: str, spec_path: Path) -> int:
    """Return the index of a spec's block of candidate seeds, numbered over every spec of every suite.

    Numbering over every spec gives a filtered run the same candidates as a full run.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML, as returned by discover_specs().
    """
    all_specs = discover_specs("all", None)
    # CPU torch generators use only the low 32 bits of a seed, so every block must end below 2**32.
    assert len(all_specs) * _SEED_STREAM_LENGTH <= 2**32, "Too many specs for separate seed streams"
    return all_specs.index((suite, spec_path))


def measure_environment(suite: str, spec_path: Path, args_cli: argparse.Namespace) -> EnvironmentRecord:
    """Return one environment spec's record after placing every seed, recording failures instead of raising.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML.
        args_cli: Parsed benchmark arguments.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.collision_mode import CollisionMode
    from isaaclab_arena.relations.passive_collision_objects import get_placement_collision_objects

    start = time.perf_counter()
    record = EnvironmentRecord(suite=suite, environment=spec_path.stem, status=EnvironmentStatus.SCORED)
    try:
        arena_env = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env()
        inputs = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg()).prepare_live_placement_inputs()
        placed_objects = [asset for asset in inputs.assets if not asset.is_anchor and asset is not arena_env.embodiment]
        if not placed_objects:
            record.status = EnvironmentStatus.SKIPPED
            record.reason = "places no objects other than anchors and the robot"
        else:
            collision_mode = None if args_cli.collision_mode is None else CollisionMode(args_cli.collision_mode)
            placement = BenchmarkPlacement(
                inputs.assets,
                inputs.placer_params,
                args_cli.num_layouts,
                get_seed_stream(suite, spec_path),
                args_cli.random_yaw_init,
                collision_mode,
            )
            collision_objects = get_placement_collision_objects(
                inputs.assets, inputs.scene_assets, placement.collision_mode
            )
            for seed in args_cli.placement_seeds:
                add_seed_results(record, placement.place(seed, collision_objects))
    except Exception as exc:
        traceback.print_exc()
        record.status = EnvironmentStatus.FAILED
        record.reason = f"{type(exc).__name__}: {exc}"
    record.wall_time_s = time.perf_counter() - start
    return record


class BenchmarkPlacement:
    """Places one environment's assets for each benchmark seed, returning every candidate."""

    def __init__(
        self,
        assets: list[PlaceableAsset],
        placer_params: ObjectPlacerParams,
        num_layouts: int,
        seed_stream: int,
        random_yaw_init: bool = False,
        collision_mode: CollisionMode | None = None,
    ):
        """Copy the placer parameters, apply the benchmark overrides and build the placer.

        Args:
            assets: Placement assets, including at least one anchor.
            placer_params: The environment's placer parameters; they are copied, not modified.
            num_layouts: Layouts to place per seed.
            seed_stream: Index of the environment's own block of candidate seeds.
            random_yaw_init: Whether to force a random initial yaw for every placed object.
            collision_mode: Default collision mode to use instead of the environment's, if given.
        """
        from isaaclab_arena.relations.object_placer import ObjectPlacer

        params = copy.copy(placer_params)
        params.solver_params = copy.copy(params.solver_params)
        params.reachability_config = copy.copy(params.reachability_config)
        if random_yaw_init:
            params.random_yaw_init = True
        if collision_mode is not None:
            params.solver_params.collision_mode = collision_mode
        # A layout normally keeps the best of max_placement_attempts candidates. Solve one candidate per attempt and
        # request all of them, so the same candidates come back ranked and the valid ones can be counted.
        self._attempts = params.max_placement_attempts
        params.max_placement_attempts = 1
        # Validators are built once here; place() only changes the placer's seed.
        self._placer = ObjectPlacer(params)
        self._assets = assets
        self._num_layouts = num_layouts
        self._seed_stream_start = seed_stream * _SEED_STREAM_LENGTH

    @property
    def collision_mode(self) -> CollisionMode:
        """Default collision mode used for placement."""
        return self._placer.params.solver_params.collision_mode

    def place(self, seed: int, collision_objects: list[CollisionObject]) -> list[list[PlacementResult]]:
        """Place num_layouts layouts for one seed and return every candidate, ranked per layout.

        The layouts match ObjectPlacer.place() with placement seed
        seed_stream * 2**20 + seed * num_layouts * max_placement_attempts. Candidates are seeded consecutively from the
        placement seed, so different seeds and different seed streams solve disjoint candidates.

        Args:
            seed: Nonnegative benchmark seed.
            collision_objects: Fixed obstacles to avoid.

        Returns:
            For each layout, its max_placement_attempts candidates, valid and lower-loss ones first.
        """
        candidates_per_seed = self._num_layouts * self._attempts
        assert (seed + 1) * candidates_per_seed <= _SEED_STREAM_LENGTH, (
            f"Seed {seed} needs {(seed + 1) * candidates_per_seed} candidate seeds, more than the {_SEED_STREAM_LENGTH}"
            " reserved per environment; use smaller seeds or fewer layouts"
        )
        self._placer.params.placement_seed = self._seed_stream_start + seed * candidates_per_seed
        return self._placer.place_ranked_per_env(
            self._assets,
            num_envs=self._num_layouts,
            results_per_env=self._attempts,
            collision_objects=collision_objects,
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
        record.candidate_successes += sum(result.success for result in ranked)
        for result in ranked:
            for check in result.validation_results.get_failed_validation_check_names:
                record.candidate_failures_per_check[check] = record.candidate_failures_per_check.get(check, 0) + 1
        validation = best.validation_results
        if best.success:
            record.layout_successes += 1
        else:
            for check in validation.get_failed_validation_check_names:
                record.failed_checks[check] = record.failed_checks.get(check, 0) + 1
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
