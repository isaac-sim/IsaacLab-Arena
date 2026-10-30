# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure relation-placement success of environment graph specs through the normal environment build.

Each spec is built once per seed index with ArenaEnvBuilder.build_env_cfg(), without stepping physics, and the
placement diagnostics of every build are added up in the spec's SpecRecord.
"""

from __future__ import annotations

import fnmatch
import subprocess
import time
import traceback
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena_benchmarking.object_placement.results import PlacementBenchmarkSettings, SpecRecord

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_diagnostics import PlacementDiagnostics

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Checkout that holds the benchmarked specs and whose commit the results record."""

SUITE_SPEC_GLOBS = {
    "robolab": "isaaclab_arena_environments/robolab/tasks/*.yaml",
    "kitchen_bench": "isaaclab_arena_environments/kitchen_bench/*.yaml",
}
"""Suite name mapped to a glob over its environment graph-spec YAML files, relative to the repository root."""

_BENCHMARKED_PACKAGES = ("isaaclab_arena", "isaaclab_arena_benchmarking", "isaaclab_arena_environments")
"""Packages whose changes can change the results, relative to the repository root."""


def discover_specs(suite: str, spec_filters: list[str] | None) -> list[tuple[str, Path]]:
    """Return (suite, spec path) pairs in a stable order, keeping specs that match any filter.

    Args:
        suite: Suite name, or "all" for every suite.
        spec_filters: Globs on spec file stems, or None to keep every spec.
    """
    suites = list(SUITE_SPEC_GLOBS) if suite == "all" else [suite]
    specs = []
    for suite_name in suites:
        for spec_path in sorted(REPO_ROOT.glob(SUITE_SPEC_GLOBS[suite_name])):
            if spec_filters is None or any(fnmatch.fnmatch(spec_path.stem, pattern) for pattern in spec_filters):
                specs.append((suite_name, spec_path))
    return specs


def get_git_provenance() -> tuple[str | None, str | None, bool | None]:
    """Return the checkout's commit, branch and whether it has uncommitted changes, or None where they are unknown.

    The branch is None for a detached checkout. Changes count when they are to tracked files or are new files in the
    packages the benchmark reads, such as a new spec. Other untracked files, such as local tool settings that only a
    user's own git config ignores, do not.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return None, None, None
    branch = _run_git("symbolic-ref", "--short", "--quiet", "HEAD")
    tracked_changes = _run_git("status", "--porcelain", "--untracked-files=no")
    package_changes = _run_git("status", "--porcelain", "--", *_BENCHMARKED_PACKAGES)
    if tracked_changes is None or package_changes is None:
        return commit, branch, None
    return commit, branch, bool(tracked_changes or package_changes)


def measure_spec(suite: str, spec_path: Path, settings: PlacementBenchmarkSettings) -> SpecRecord:
    """Return one spec's record after building it once per seed index, recording failures instead of raising.

    Each build is the normal ArenaEnvBuilder.build_env_cfg() for settings.num_envs environments, without stepping
    physics; the record adds up the placement diagnostics of every build. A spec whose build solves no relations
    fails, since it has no placement to measure.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML.
        settings: Benchmark settings.
    """
    start = time.perf_counter()
    record = SpecRecord(suite=suite, spec=spec_path.stem)
    try:
        for seed_index in settings.seed_indices:
            diagnostics = _measure_build(suite, spec_path, settings, seed_index)
            record.add_build(diagnostics, settings.num_envs)
    except Exception as exc:
        traceback.print_exc()
        record.error = f"{type(exc).__name__}: {exc}"
    record.wall_time_s = time.perf_counter() - start
    return record


def get_seeds_per_build(placer_params: ObjectPlacerParams, num_envs: int) -> int:
    """Return the most candidate seeds one build can use.

    A placement pool solves at most max_placement_attempts batches, each of at most
    num_envs * min_unique_layouts_per_env * max_placement_attempts candidates, seeded consecutively from the
    placement seed.

    Args:
        placer_params: The placer parameters the build uses.
        num_envs: Environments per build.
    """
    attempts = placer_params.max_placement_attempts
    return attempts * num_envs * placer_params.min_unique_layouts_per_env * attempts


def get_placement_seed(suite: str, spec_stem: str, seed_index: int, seeds_per_build: int) -> int:
    """Return the placement seed of one build of a spec.

    A spec's builds take consecutive ranges of candidate seeds, starting from a hash of its suite and file stem, so
    adding or removing other specs leaves its candidates unchanged. Different specs' ranges are apart unless their
    hashes happen to fall within a few ranges of each other.

    Args:
        suite: Suite the spec belongs to.
        spec_stem: Spec file stem.
        seed_index: Nonnegative index of the build among the spec's builds.
        seeds_per_build: Candidate seeds reserved for each build, from get_seeds_per_build().
    """
    return zlib.crc32(f"{suite}/{spec_stem}".encode()) + seed_index * seeds_per_build


def _measure_build(
    suite: str, spec_path: Path, settings: PlacementBenchmarkSettings, seed_index: int
) -> PlacementDiagnostics:
    """Build a spec once, for one seed index, and return the build's placement diagnostics.

    Args:
        suite: Suite the spec belongs to.
        spec_path: Environment graph-spec YAML.
        settings: Benchmark settings.
        seed_index: Index of the build among the spec's builds.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    # A build applies build-time variations to its environment, so every build starts from the spec.
    arena_env = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env()
    # Loading a spec creates new placer parameters, built from the defaults and the spec's overrides.
    placer_params = arena_env.placer_params
    assert placer_params is not None, f"{spec_path} has no placer parameters"
    if settings.random_yaw_init is not None:
        placer_params.random_yaw_init = settings.random_yaw_init
    if settings.collision_mode is not None:
        placer_params.solver_params.collision_mode = settings.collision_mode
    seeds_per_build = get_seeds_per_build(placer_params, settings.num_envs)
    placement_seed = get_placement_seed(suite, spec_path.stem, seed_index, seeds_per_build)
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=settings.num_envs, placement_seed=placement_seed))
    builder.build_env_cfg()
    diagnostics = builder.placement_diagnostics
    assert diagnostics is not None, "the build solves no relations"
    # Each candidate takes one seed, so a build that solves more than it reserved overlaps the next build.
    num_seeds_used = sum(batch.num_candidates for batch in diagnostics.batches)
    assert (
        num_seeds_used <= seeds_per_build
    ), f"The build used {num_seeds_used} candidate seeds, more than the {seeds_per_build} reserved for it"
    return diagnostics


def _run_git(*args: str) -> str | None:
    """Return git's stripped output in the repository, or None when git is unavailable or fails."""
    # Trust the checkout even when it belongs to another user, as it does for root inside the container.
    command = ["git", "-c", f"safe.directory={REPO_ROOT}", *args]
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None
