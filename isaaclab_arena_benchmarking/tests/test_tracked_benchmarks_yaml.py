# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that the tracked benchmark list ``isaaclab_arena_benchmarking/benchmark_lists/tracked_benchmarks.yaml`` is well formed."""

import re
import subprocess
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from osmo.submit_arena_experiment import build_arena_experiment_submission_cfg

REPO_ROOT = Path(TestConstants.repo_root).resolve()
TRACKED_BENCHMARKS_PATH = REPO_ROOT / "isaaclab_arena_benchmarking" / "benchmark_lists" / "tracked_benchmarks.yaml"

BENCHMARK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
"""Benchmark names are used in OSMO workflow names and folder paths."""

BENCHMARK_ENTRY_KEYS = {"name", "experiment_cfg", "overrides"}

REQUIRED_BENCHMARK_ENTRY_KEYS = {"name", "experiment_cfg"}

RESERVED_OVERRIDE_KEYS = ("osmo.workflow_name", "osmo.output_url", "experiment_cfg.name")
"""OSMO settings chosen per launch, which a benchmark entry must not fix."""


def _load_benchmark_entries() -> list[dict]:
    content = yaml.safe_load(TRACKED_BENCHMARKS_PATH.read_text(encoding="utf-8"))
    assert isinstance(content, dict) and set(content) == {"benchmarks"}, "expected exactly one 'benchmarks' key"
    entries = content["benchmarks"]
    assert isinstance(entries, list) and entries, "'benchmarks' must be a non-empty list"
    assert all(isinstance(entry, dict) for entry in entries), "each benchmark must be a mapping"
    return entries


BENCHMARK_ENTRIES = _load_benchmark_entries()


def _entry_id(entry: dict) -> str:
    return str(entry.get("name"))


def test_benchmark_names_are_unique():
    names = [entry.get("name") for entry in BENCHMARK_ENTRIES]
    assert len(names) == len(set(names)), f"duplicate benchmark names in {names}"


@pytest.mark.parametrize("entry", BENCHMARK_ENTRIES, ids=_entry_id)
def test_benchmark_entry_is_well_formed(entry):
    name = entry.get("name")
    assert isinstance(name, str) and BENCHMARK_NAME_PATTERN.match(
        name
    ), f"benchmark name {name!r} must match {BENCHMARK_NAME_PATTERN.pattern}"
    unknown_keys = set(entry) - BENCHMARK_ENTRY_KEYS
    assert not unknown_keys, f"benchmark {name!r} has unknown keys {sorted(unknown_keys)}"
    missing_keys = REQUIRED_BENCHMARK_ENTRY_KEYS - set(entry)
    assert not missing_keys, f"benchmark {name!r} is missing required keys {sorted(missing_keys)}"

    experiment_cfg = entry.get("experiment_cfg")
    assert isinstance(experiment_cfg, str), f"benchmark {name!r} needs a string experiment_cfg"
    assert not Path(experiment_cfg).is_absolute(), f"benchmark {name!r} experiment_cfg must be relative to the repo"
    experiment_cfg_path = (REPO_ROOT / experiment_cfg).resolve()
    assert experiment_cfg_path.is_relative_to(REPO_ROOT), f"benchmark {name!r} experiment_cfg is outside the repo"
    assert experiment_cfg_path.is_file(), f"benchmark {name!r} experiment_cfg {experiment_cfg!r} does not exist"

    overrides = entry.get("overrides", [])
    assert isinstance(overrides, list) and all(
        isinstance(override, str) for override in overrides
    ), f"benchmark {name!r} overrides must be a list of strings"
    for override in overrides:
        # Hydra prefixes: '+' adds, '++' forces, '~' deletes a key.
        override_key = override.lstrip("+~").split("=", 1)[0]
        assert (
            override_key not in RESERVED_OVERRIDE_KEYS
        ), f"benchmark {name!r} override {override!r} sets reserved key {override_key!r}"


def _compose_benchmark_submission(entry: dict):
    return build_arena_experiment_submission_cfg(
        REPO_ROOT / entry["experiment_cfg"],
        [*entry.get("overrides", []), f"experiment_cfg.name='{entry['name']}'"],
    )


@pytest.mark.parametrize("entry", BENCHMARK_ENTRIES, ids=_entry_id)
def test_benchmark_experiment_exports_kpis(entry):
    """Every tracked Experiment selects its KPI exporters explicitly, not the config default."""
    experiment_cfg = _compose_benchmark_submission(entry).experiment_cfg

    expected_exporters = (
        ["placement_success", "timing"] if entry["name"] == "placement-subset" else ["success_rate", "timing"]
    )
    assert experiment_cfg.kpi_exporters == expected_exporters, f"benchmark {entry['name']!r}"


def test_smoke_single_run_benchmark_keeps_only_the_baseline_run():
    entry = next(entry for entry in BENCHMARK_ENTRIES if entry["name"] == "smoke-single-run")

    experiment_cfg = _compose_benchmark_submission(entry).experiment_cfg

    assert list(experiment_cfg.runs) == ["baseline"]


@pytest.mark.parametrize("entry", BENCHMARK_ENTRIES, ids=_entry_id)
def test_benchmark_renders_an_osmo_workflow_dry_run(entry):
    result = subprocess.run(
        [
            TestConstants.python_path,
            "-m",
            "osmo.submit_arena_experiment",
            "--experiment_cfg",
            entry["experiment_cfg"],
            "--dry_run",
            *entry.get("overrides", []),
            # The per-launch setting an internal launcher adds from the entry.
            f"experiment_cfg.name='{entry['name']}'",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"dry run of {entry['name']!r} failed:\n{result.stdout}\n{result.stderr}"
    assert "[dry-run] Rendered workflow YAML:" in result.stdout, f"no rendered workflow for {entry['name']!r}"
    assert "\nworkflow:\n" in result.stdout, f"rendered output for {entry['name']!r} has no workflow section"
