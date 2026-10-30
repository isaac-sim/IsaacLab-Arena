# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement success benchmark: measuring specs, seeding builds, provenance and results files."""

import json
import subprocess

import pytest

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_diagnostics import (
    BatchValidationDiagnostics,
    PlacementDiagnostics,
    ValidationCheckDiagnostics,
)
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.tests.test_placement_diagnostics import _desk_and_box, _placer_params
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_benchmarking.object_placement import measurement
from isaaclab_arena_benchmarking.object_placement.measurement import (
    discover_specs,
    get_placement_seed,
    get_seeds_per_build,
)
from isaaclab_arena_benchmarking.object_placement.results import (
    PlacementBenchmarkResults,
    PlacementBenchmarkSettings,
    SpecRecord,
    format_results,
    summarize_suites,
)


# Defined first so that, when this file runs alone, the SimulationApp starts before later tests solve placements.
def _test_real_spec_builds_once_per_seed_index_and_a_missing_spec_fails(simulation_app) -> bool:
    from unittest import mock

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_benchmarking.object_placement.measurement import measure_spec
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    settings = PlacementBenchmarkSettings(num_envs=2, seed_indices=[0, 1])
    [(_, spec_path)] = discover_specs("robolab", ["banana_on_plate"])
    placement_seeds = []
    build_without_recording = ArenaEnvBuilder.__init__

    def build_and_record_placement_seed(builder, arena_env, cfg, *args, **kwargs):
        placement_seeds.append(cfg.placement_seed)
        build_without_recording(builder, arena_env, cfg, *args, **kwargs)

    with mock.patch.object(ArenaEnvBuilder, "__init__", build_and_record_placement_seed):
        scored = measure_spec("robolab", spec_path, settings)
    failed = measure_spec("robolab", spec_path.with_name("missing_spec.yaml"), settings)

    assert scored.error is None, scored.error
    # Two normal builds of two environments, read from their placement diagnostics.
    assert (scored.num_builds, scored.num_envs) == (2, 4) and scored.num_candidates > 0
    assert scored.checks["on_relation"].num_evaluated == scored.num_candidates
    # The spec sets no placer parameters of its own, so the defaults decide the seeds each build reserves.
    seeds_per_build = get_seeds_per_build(ObjectPlacerParams(), settings.num_envs)
    assert placement_seeds == [
        get_placement_seed("robolab", "banana_on_plate", seed_index, seeds_per_build) for seed_index in (0, 1)
    ]
    assert failed.error
    return True


def test_real_spec_builds_once_per_seed_index_and_a_missing_spec_fails():
    """A spec is built once per seed index, each from its own placement seed, and a spec that cannot be built fails."""
    assert run_function_with_persistent_simulation_app(
        _test_real_spec_builds_once_per_seed_index_and_a_missing_spec_fails
    )


def test_results_file_round_trips_and_reports_suite_rates(tmp_path):
    """Suites report scored and failed counts, and equal-weight mean rates only when no spec failed."""
    results = _results([
        _scored("robolab", "pick_apple", envs=(45, 50), candidates=(300, 500), batches=6),
        _scored("robolab", "pick_banana", envs=(25, 25), candidates=(250, 250), batches=3),
        _scored("kitchen_bench", "pick_cup", envs=(50, 50), candidates=(500, 500), batches=3),
        SpecRecord("kitchen_bench", "place_lime", error="KeyError: 'lime'"),
    ])
    results.records[0].checks = {
        "no_overlap": ValidationCheckDiagnostics(500, 350, required=True),
        "face_to": ValidationCheckDiagnostics(350, 300, required=False),
    }
    results.settings = PlacementBenchmarkSettings(random_yaw_init=True, collision_mode=CollisionMode.MESH)
    path = tmp_path / "run.json"
    results.save(path)

    assert PlacementBenchmarkResults.load(path) == results
    # A file written by another version names its unknown fields instead of failing deep inside the loader.
    old_version = json.loads(path.read_text())
    old_version["records"][0]["num_layouts"] = 50
    path.write_text(json.dumps(old_version))
    with pytest.raises(AssertionError, match=r"SpecRecord fields .*\['num_layouts'\]"):
        PlacementBenchmarkResults.load(path)
    robolab, kitchen_bench = summarize_suites(results)
    # Every spec weighs the same, whatever its sample size.
    assert robolab.valid_candidate_rate == pytest.approx(0.8)
    assert robolab.valid_env_rate == pytest.approx(0.95)
    assert robolab.batches_per_build == pytest.approx(1.5)
    assert (kitchen_bench.num_scored, kitchen_bench.num_failed) == (1, 1)
    # A failed spec could have lowered the suite's rates, so they are left out.
    assert kitchen_bench.valid_candidate_rate is None
    assert "| kitchen_bench | 1 | 1 | n/a | n/a |" in format_results(results)


def test_record_counts_the_first_pool_batch_of_each_build():
    """Candidates and checks come from each build's first pool batch; later batches add only to the batch count."""
    # IK runs only on the candidates that pass no_overlap, so it rejects 10 of 70 rather than 40 of 100. face_to is
    # outside required_checks, so the record keeps it as optional.
    first_batch = BatchValidationDiagnostics(
        100,
        60,
        {
            "no_overlap": ValidationCheckDiagnostics(100, 70, True),
            "ik_reachable": ValidationCheckDiagnostics(70, 60, True),
            "face_to": ValidationCheckDiagnostics(100, 50, False),
        },
    )
    later_batch = BatchValidationDiagnostics(
        20,
        15,
        {
            "no_overlap": ValidationCheckDiagnostics(20, 16, True),
            "ik_reachable": ValidationCheckDiagnostics(16, 15, True),
            "face_to": ValidationCheckDiagnostics(20, 10, False),
        },
    )
    record = SpecRecord("kitchen_bench", "open_fridge")

    record.add_build(PlacementDiagnostics([first_batch, later_batch], fallback_envs=[3]), num_envs=10)
    record.add_build(PlacementDiagnostics([first_batch]), num_envs=10)

    assert (record.num_valid_candidates, record.num_candidates) == (120, 200)
    assert (record.num_fallback_envs, record.num_envs, record.num_batches, record.num_builds) == (1, 20, 3, 2)
    assert record.checks == {
        "no_overlap": ValidationCheckDiagnostics(200, 140, True),
        "ik_reachable": ValidationCheckDiagnostics(140, 120, True),
        "face_to": ValidationCheckDiagnostics(200, 100, False),
    }
    # Adding the same first batch twice must not change it.
    assert first_batch.checks["no_overlap"] == ValidationCheckDiagnostics(100, 70, True)


def test_builds_of_a_spec_take_adjacent_seed_ranges_that_a_build_cannot_exceed():
    """Seed indices start seeds_per_build apart, and a pool that solves every batch uses exactly that many seeds."""
    params = _placer_params()
    num_envs = 2
    seeds_per_build = get_seeds_per_build(params, num_envs)

    # A box wider than the desk is never valid. The pool has the size an environment build gives it.
    pool_size = num_envs * params.min_unique_layouts_per_env
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=1.5), params, pool_size=pool_size, num_envs=num_envs)

    batches = pool.last_fill_diagnostics.batches
    assert len(batches) == params.max_placement_attempts
    assert sum(batch.num_candidates for batch in batches) == seeds_per_build
    first_build, second_build = (
        get_placement_seed("robolab", "box_on_desk", index, seeds_per_build) for index in (0, 1)
    )
    assert second_build - first_build == seeds_per_build


def test_git_provenance_reads_the_commit_branch_and_benchmarked_changes(tmp_path, monkeypatch):
    """Provenance names the commit and branch, and counts new files only inside the benchmarked packages."""
    monkeypatch.setattr(measurement, "REPO_ROOT", tmp_path)
    # Keep git from finding a repository that encloses the temporary folder.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert measurement.get_git_provenance() == (None, None, None)

    _git(tmp_path, "init", "--initial-branch", "main")
    (tmp_path / "README.md").write_text("Benchmarked checkout\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "--message", "Add README")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    assert measurement.get_git_provenance() == (commit, "main", False)
    (tmp_path / "notes.txt").write_text("Local notes\n")
    assert measurement.get_git_provenance() == (commit, "main", False)
    (tmp_path / "isaaclab_arena_environments").mkdir()
    (tmp_path / "isaaclab_arena_environments" / "new_spec.yaml").write_text("")
    assert measurement.get_git_provenance() == (commit, "main", True)
    # A detached checkout has no branch.
    _git(tmp_path, "checkout", "--detach")
    assert measurement.get_git_provenance() == (commit, None, True)


def test_spec_discovery_filters_by_stem():
    """Every suite contributes specs, and a stem glob keeps only matching specs."""
    specs = discover_specs("all", None)
    assert {suite for suite, _ in specs} == {"robolab", "kitchen_bench"}
    stem = specs[0][1].stem
    assert [path.stem for _, path in discover_specs("all", [stem])] == [stem]


def test_specs_seed_ranges_are_apart_at_the_default_settings():
    """At the default settings, no two specs' candidate seeds overlap, so specs that share a scene sample it apart."""
    # No spec sets its own batch parameters, so every spec reserves the default number of seeds per build.
    settings = PlacementBenchmarkSettings()
    seeds_per_build = get_seeds_per_build(ObjectPlacerParams(), settings.num_envs)
    first_seeds = sorted(
        get_placement_seed(suite, path.stem, 0, seeds_per_build) for suite, path in discover_specs("all", None)
    )
    # The candidate generator seeds a CPU generator, which keeps only the low 32 bits, so seeds wrap at 2**32.
    first_seeds.append(first_seeds[0] + 2**32)
    seeds_per_spec = len(settings.seed_indices) * seeds_per_build
    assert all(next_seed - seed >= seeds_per_spec for seed, next_seed in zip(first_seeds, first_seeds[1:]))


def _git(repository, *args: str) -> str:
    """Run git in repository as a test author and return its stripped output."""
    command = ["git", "-c", "user.name=Arena Test", "-c", "user.email=arena-test@example.com"]
    command += ["-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, cwd=repository, capture_output=True, text=True, check=True).stdout.strip()


def _scored(suite: str, spec: str, envs: tuple[int, int], candidates: tuple[int, int], batches: int) -> SpecRecord:
    """Return a scored record from (valid, total) environment and candidate counts, over three builds."""
    return SpecRecord(
        suite,
        spec,
        num_builds=3,
        num_candidates=candidates[1],
        num_valid_candidates=candidates[0],
        num_envs=envs[1],
        num_fallback_envs=envs[1] - envs[0],
        num_batches=batches,
    )


def _results(records: list[SpecRecord]) -> PlacementBenchmarkResults:
    return PlacementBenchmarkResults(
        start_time="2026-10-21T00:00:00+00:00",
        commit="abc123",
        branch="main",
        dirty=False,
        gpu_name="NVIDIA RTX 6000 Ada Generation",
        settings=PlacementBenchmarkSettings(),
        selected_specs=[record.spec for record in records],
        records=records,
    )
