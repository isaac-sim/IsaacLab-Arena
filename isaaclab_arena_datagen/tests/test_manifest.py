# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone datagen manifest module (no Isaac Sim required)."""

import json
import os

from isaaclab_arena_datagen import manifest as m


def _sequence_dict(idx=0, outcome="success"):
    return {
        "episode_index": idx,
        "path": f"/data/job_a/episode_{idx:04d}/dataset.h5",
        "num_frames": 100 + idx,
        "camera_ids": ["cam0"],
        "dynamic_object_names": ["rigid_object_1_lemon"],
        "outcome": outcome,
    }


def test_clean_datagen_settings_drops_output_dir():
    cleaned = m.clean_datagen_settings({"output_dir": "/x", "width": 640})
    assert cleaned == {"width": 640}


def test_relativize_path_under_root():
    assert m.relativize_path("/data/job_a/episode_0000/dataset.h5", "/data") == "job_a/episode_0000/dataset.h5"


def test_relativize_path_outside_root_stays_absolute():
    assert m.relativize_path("/other/x.h5", "/data") == "/other/x.h5"


def test_build_job_record_counts_and_relativizes():
    job = m.build_job_record(
        name="job_a",
        status="completed",
        policy_type="pkg.Policy",
        policy_config={"k": 1},
        language_instruction="pick it up",
        arena_env_args=["--environment", "e"],
        datagen_settings={"width": 640},
        sim=m.SimInfo(dt=0.005),
        sequence_dicts=[_sequence_dict(0, "success"), _sequence_dict(1, "timeout")],
        root="/data",
    )
    assert job.num_sequences == 2
    assert job.num_success == 1
    assert job.sequences[0].path == "job_a/episode_0000/dataset.h5"
    assert job.sequences[1].outcome == "timeout"


def test_capture_git_info_returns_sha_in_repo():
    info = m.capture_git_info(os.path.dirname(os.path.abspath(__file__)))
    assert info.sha is not None and len(info.sha) >= 7


def test_capture_system_info_never_raises_and_has_python():
    info = m.capture_system_info(device="cuda:0")
    assert info.python_version is not None
    assert info.device == "cuda:0"


def test_build_and_write_manifest_round_trip(tmp_path):
    job = m.build_job_record(
        name="job_a",
        status="completed",
        policy_type="pkg.Policy",
        policy_config={},
        language_instruction=None,
        arena_env_args=[],
        datagen_settings={"width": 640},
        sim=m.SimInfo(dt=0.005, render_carb_settings={"a": 1}),
        sequence_dicts=[_sequence_dict()],
        root="/data",
    )
    manifest = m.build_manifest(
        created_at="2026-06-26T00:00:00Z",
        description="why",
        generator_tool="eval_runner",
        git=m.GitInfo(sha="abc1234", branch="main", dirty=False),
        system=m.capture_system_info("cuda:0"),
        input_config={"jobs": []},
        jobs=[job],
    )
    path = str(tmp_path / "manifest.json")
    assert m.write_manifest(path, manifest) is True
    assert not os.path.exists(path + ".tmp")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["description"] == "why"
    assert data["jobs"][0]["num_success"] == 1
    assert data["jobs"][0]["sequences"][0]["path"] == "job_a/episode_0000/dataset.h5"


def test_write_manifest_returns_false_on_bad_path_without_raising():
    manifest = m.build_manifest(
        created_at="t",
        description=None,
        generator_tool="t",
        git=m.GitInfo(),
        system=m.SystemInfo(),
        input_config=None,
        jobs=[],
    )
    # A path whose parent is a file (not a dir) cannot be created.
    assert m.write_manifest("/dev/null/nope/manifest.json", manifest) is False
