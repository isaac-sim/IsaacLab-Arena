# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from osmo.benchmarks.benchmark_list import BenchmarkEntry, load_benchmark_list

MAIN_BENCHMARKS_PATH = Path(TestConstants.repo_root) / "osmo" / "benchmarks" / "main_benchmarks.yaml"
EXISTING_CFG = "isaaclab_arena_environments/experiment_configs/droid_pnp_openpi_experiment.yaml"


def _write_list(tmp_path: Path, content: dict) -> Path:
    path = tmp_path / "benchmarks.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


def _entry(name: str = "a", experiment_cfg: str = EXISTING_CFG, **extra) -> dict:
    return {"name": name, "experiment_cfg": experiment_cfg, **extra}


def test_load_valid_list(tmp_path):
    path = _write_list(
        tmp_path,
        {"benchmarks": [_entry("first"), _entry("second-2", overrides=["osmo.pool=x"])]},
    )
    entries = load_benchmark_list(path)
    assert entries == [
        BenchmarkEntry(name="first", experiment_cfg=EXISTING_CFG, overrides=[]),
        BenchmarkEntry(name="second-2", experiment_cfg=EXISTING_CFG, overrides=["osmo.pool=x"]),
    ]


@pytest.mark.parametrize(
    "override",
    ["osmo.workflow_name=x", "+osmo.workflow_name=x", "osmo.output_url=y", "~osmo.output_url", "++osmo.output_url=z"],
)
def test_rejects_reserved_overrides(tmp_path, override):
    path = _write_list(tmp_path, {"benchmarks": [_entry("a", overrides=[override])]})
    with pytest.raises(AssertionError, match="reserved key"):
        load_benchmark_list(path)


def test_rejects_duplicate_names(tmp_path):
    path = _write_list(tmp_path, {"benchmarks": [_entry("dup"), _entry("dup")]})
    with pytest.raises(AssertionError, match="dup"):
        load_benchmark_list(path)


@pytest.mark.parametrize("name", ["Has Space", "UPPER", "a/b", "-lead", ""])
def test_rejects_unsafe_names(tmp_path, name):
    path = _write_list(tmp_path, {"benchmarks": [_entry(name)]})
    with pytest.raises(AssertionError, match="must match"):
        load_benchmark_list(path)


def test_rejects_missing_experiment_file(tmp_path):
    path = _write_list(tmp_path, {"benchmarks": [_entry("gone", "does/not/exist.yaml")]})
    with pytest.raises(AssertionError, match="gone"):
        load_benchmark_list(path)


def test_rejects_experiment_file_outside_repo(tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("x: 1\n")
    repo_root = Path(TestConstants.repo_root).resolve()
    traversing = Path(os.path.relpath(outside, repo_root)).as_posix()
    assert traversing.startswith("..")
    for experiment_cfg in (traversing, str(outside)):
        path = _write_list(tmp_path, {"benchmarks": [_entry("escape", experiment_cfg)]})
        with pytest.raises(AssertionError, match="escape"):
            load_benchmark_list(path)


def test_rejects_empty_list(tmp_path):
    path = _write_list(tmp_path, {"benchmarks": []})
    with pytest.raises(AssertionError):
        load_benchmark_list(path)


def test_rejects_unknown_keys(tmp_path):
    with pytest.raises(AssertionError, match="extra"):
        load_benchmark_list(_write_list(tmp_path, {"benchmarks": [_entry("a")], "extra": 1}))
    with pytest.raises(AssertionError, match="extra"):
        load_benchmark_list(_write_list(tmp_path, {"benchmarks": [_entry("a", extra=1)]}))


def test_rejects_non_string_overrides(tmp_path):
    path = _write_list(tmp_path, {"benchmarks": [_entry("a", overrides=[1])]})
    with pytest.raises(AssertionError, match="a"):
        load_benchmark_list(path)


def test_main_benchmarks_yaml_is_valid():
    entries = load_benchmark_list(MAIN_BENCHMARKS_PATH)
    assert len(entries) >= 1


@pytest.mark.parametrize("entry", load_benchmark_list(MAIN_BENCHMARKS_PATH), ids=lambda e: e.name)
def test_every_entry_renders_a_dry_run(entry):
    result = subprocess.run(
        [
            TestConstants.python_path,
            "-m",
            "osmo.submit_arena_experiment",
            "--experiment_cfg",
            entry.experiment_cfg,
            "--dry_run",
            *entry.overrides,
        ],
        cwd=TestConstants.repo_root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"dry run of '{entry.name}' failed:\n{result.stdout}\n{result.stderr}"
    assert "[dry-run] Rendered workflow YAML:" in result.stdout, f"no rendered workflow for '{entry.name}'"
    assert "\nworkflow:\n" in result.stdout, f"rendered output for '{entry.name}' has no workflow section"
