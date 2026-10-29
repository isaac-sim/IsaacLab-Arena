# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success of environment graph specs through the normal environment build.

Each spec is built once per placement seed with ArenaEnvBuilder.build_env_cfg(), without stepping physics, and the
placement diagnostics of every build are added up in the spec's EnvironmentRecord.
"""

from __future__ import annotations

import copy
import fnmatch
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_diagnostics import ValidationCheckDiagnostics
from isaaclab_arena_benchmarking.object_placement.results import (
    EnvironmentRecord,
    EnvironmentStatus,
    PlacementBenchmarkSettings,
)

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_diagnostics import PlacementDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Checkout that holds the benchmarked specs and whose commit the results record."""

SUITE_SPEC_GLOBS = {
    "robolab": "isaaclab_arena_environments/robolab/tasks/*.yaml",
    "kitchen_bench": "isaaclab_arena_environments/kitchen_bench/*.yaml",
}
"""Suite name mapped to a glob over its environment graph-spec YAML files, relative to the repository root.

Append new suites, so that the specs of earlier suites keep their seed blocks.
"""

_BENCHMARKED_PACKAGES = ("isaaclab_arena", "isaaclab_arena_benchmarking", "isaaclab_arena_environments")
"""Packages whose changes can change the results, relative to the repository root."""

_STAMPED_COMMIT_ENV_VAR = "ARENA_GIT_SHA"
"""Environment variable an Arena image can set to its commit, with a "-dirty" suffix for uncommitted changes."""

_STAMPED_BRANCH_ENV_VAR = "ARENA_GIT_BRANCH"
"""Environment variable an Arena image can set to its branch."""

_UNKNOWN_STAMP_VALUE = "unknown"
"""Value an Arena image built without its commit or branch stamps into those variables."""

_SEEDS_PER_ENVIRONMENT = 2**20
"""Size of each environment's block of candidate seeds, so environments never share candidates, even when they share
a scene."""


def discover_specs(suite: str, env_filters: list[str] | None) -> list[tuple[str, Path]]:
    """Return (suite, spec path) pairs in a stable order, keeping specs that match any filter.

    Args:
        suite: Suite name, or "all" for every suite.
        env_filters: Globs on spec file stems, or None to keep every spec.
    """
    suites = list(SUITE_SPEC_GLOBS) if suite == "all" else [suite]
    specs = []
    for suite_name in suites:
        for spec_path in sorted(REPO_ROOT.glob(SUITE_SPEC_GLOBS[suite_name])):
            if env_filters is None or any(fnmatch.fnmatch(spec_path.stem, pattern) for pattern in env_filters):
                specs.append((suite_name, spec_path))
    return specs


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


def get_git_provenance() -> tuple[str | None, str | None, bool | None]:
    """Return the checkout's commit, branch and whether it has uncommitted changes, or None where they are unknown.

    Changes count when they are to tracked files or are new files in the packages the benchmark reads, such as a new
    spec. Other untracked files, such as local tool settings that only a user's own git config ignores, do not. When
    git cannot read the checkout, as in an image without .git, the values come from the ARENA_GIT_SHA and
    ARENA_GIT_BRANCH environment variables when they hold a value other than "unknown".
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        stamped_commit = _read_stamp(_STAMPED_COMMIT_ENV_VAR)
        if stamped_commit is None:
            return None, None, None
        stamped_branch = _read_stamp(_STAMPED_BRANCH_ENV_VAR)
        return stamped_commit.removesuffix("-dirty"), stamped_branch, stamped_commit.endswith("-dirty")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    tracked_changes = _run_git("status", "--porcelain", "--untracked-files=no")
    package_changes = _run_git("status", "--porcelain", "--", *_BENCHMARKED_PACKAGES)
    if tracked_changes is None or package_changes is None:
        return commit, branch, None
    return commit, branch, bool(tracked_changes or package_changes)


def measure_environment(
    suite: str, spec_path: Path, settings: PlacementBenchmarkSettings, seed_block_index: int
) -> EnvironmentRecord:
    """Return one environment spec's record after building it once per seed, recording failures instead of raising.

    Each build is the normal ArenaEnvBuilder.build_env_cfg() for settings.num_envs environments, without stepping
    physics; the record adds up the placement diagnostics of every build.

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
    """Add one build's placement diagnostics to the environment record.

    Candidate and check counts come from the pool's first batch, which has the same size in every build; later
    batches only refill environments that lacked valid layouts, so including them would favor scenes that need more
    batches.

    Args:
        record: Record to update in place.
        diagnostics: Placement diagnostics of the build.
        num_envs: Environments in the build.
    """
    first_batch = diagnostics.batches[0]
    required_per_check = _required_per_check(first_batch.checks)
    assert (
        record.num_builds == 0 or _required_per_check(record.checks) == required_per_check
    ), f"Checks differ between builds of {record.environment}"
    record.num_builds += 1
    record.num_envs += num_envs
    record.num_batches += len(diagnostics.batches)
    record.num_fallback_envs += len(set(diagnostics.fallback_envs))
    record.num_candidates += first_batch.num_candidates
    record.num_valid_candidates += first_batch.num_valid
    for check, check_diagnostics in first_batch.checks.items():
        # Sum into the record's own counts, leaving the build's diagnostics unchanged.
        check_totals = record.checks.setdefault(check, ValidationCheckDiagnostics(0, 0, check_diagnostics.required))
        check_totals.num_evaluated += check_diagnostics.num_evaluated
        check_totals.num_passed += check_diagnostics.num_passed


def _required_per_check(checks: dict[str, ValidationCheckDiagnostics]) -> dict[str, bool]:
    return {check: check_diagnostics.required for check, check_diagnostics in checks.items()}


def _read_stamp(env_var: str) -> str | None:
    """Return the git value an Arena image stamped into env_var, or None when it is unset, empty or "unknown"."""
    value = os.environ.get(env_var)
    return None if value in (None, "", _UNKNOWN_STAMP_VALUE) else value


def _run_git(*args: str) -> str | None:
    """Return git's stripped output in the repository, or None when git is unavailable or fails."""
    # Trust the checkout even when it belongs to another user, as it does for root inside the container.
    command = ["git", "-c", f"safe.directory={REPO_ROOT}", *args]
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None
