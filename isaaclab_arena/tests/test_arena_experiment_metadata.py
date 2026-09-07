# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Experiment Runner metadata artifact."""

import hashlib
import json
from types import SimpleNamespace

import isaaclab_arena.evaluation.arena_experiment_metadata as metadata_module
from isaaclab_arena.evaluation.arena_experiment_metadata import (
    ARENA_EXPERIMENT_METADATA_FILENAME,
    ARENA_EXPERIMENT_METADATA_SCHEMA_VERSION,
    ArenaExperimentMetadataRecorder,
)


def _runner_arguments() -> SimpleNamespace:
    """Return the runner fields captured by the metadata recorder."""
    return SimpleNamespace(
        device="cuda:1",
        headless=True,
        visualizer="none",
        rendering_mode="balanced",
        enable_cameras=True,
        record_viewport_video=False,
        record_camera_video=True,
    )


def _resolved_experiment(camera_height=360, camera_width=640) -> SimpleNamespace:
    """Return a minimal resolved Experiment configuration for metadata tests."""
    run_cfg = SimpleNamespace(
        environment=SimpleNamespace(enable_cameras=True),
        policy=SimpleNamespace(),
        environment_builder=SimpleNamespace(
            num_envs=64,
            device="cuda:1",
            seed=7,
            placement_seed=11,
            camera_height=camera_height,
            camera_width=camera_width,
        ),
        rollout_limit=SimpleNamespace(num_steps=500, num_episodes=None),
        num_rebuilds=3,
        variations={"object": {"mass": {"enabled": False}}},
    )
    return SimpleNamespace(runs={"camera_baseline": run_cfg})


def test_metadata_recorder_persists_invocation_and_resolved_run(tmp_path, monkeypatch):
    """Write the initial artifact early, then enrich it with resolved benchmark fields."""
    monkeypatch.setattr(
        metadata_module,
        "_resolve_source_revision",
        lambda: {
            "git_commit": "arena-commit",
            "git_dirty": False,
            "isaaclab_submodule": {
                "git_commit": "isaaclab-commit",
                "expected_git_commit": "isaaclab-commit",
                "matches_superproject": True,
                "git_dirty": False,
            },
        },
    )
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_contents = "runs: {}\n"
    experiment_config_path.write_text(experiment_config_contents, encoding="utf-8")
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    command = ["python", "experiment_runner.py", "runs.camera_baseline.environment_builder.num_envs=64"]

    recorder = ArenaExperimentMetadataRecorder.start(
        output_directory,
        experiment_config_path,
        ["runs.camera_baseline.environment_builder.num_envs=64"],
        _runner_arguments(),
        command=command,
    )
    metadata_path = output_directory / ARENA_EXPERIMENT_METADATA_FILENAME
    initial_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert initial_metadata["schema_version"] == ARENA_EXPERIMENT_METADATA_SCHEMA_VERSION
    assert initial_metadata["status"] == "starting"
    assert initial_metadata["experiment_config"] == {
        "path": str(experiment_config_path.resolve()),
        "format": "yaml",
        "sha256": hashlib.sha256(experiment_config_contents.encode()).hexdigest(),
    }
    assert initial_metadata["experiment_output_directory"] == str(output_directory.resolve())
    assert initial_metadata["command"] == command
    assert initial_metadata["experiment_overrides"] == ["runs.camera_baseline.environment_builder.num_envs=64"]
    assert initial_metadata["runtime"]["rendering_mode"] == "balanced"
    assert initial_metadata["runtime"]["environment_render_mode"] is None
    assert initial_metadata["revision"]["isaaclab_submodule"]["matches_superproject"] is True
    assert initial_metadata["runs"] == {}

    recorder.record_resolved_experiment(_resolved_experiment())
    resolved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_metadata = resolved_metadata["runs"]["camera_baseline"]

    assert resolved_metadata["status"] == "running"
    assert resolved_metadata["resolved_at"] is not None
    assert run_metadata["num_envs"] == 64
    assert run_metadata["device"] == "cuda:1"
    assert run_metadata["seed"] == 7
    assert run_metadata["placement_seed"] == 11
    assert run_metadata["camera"] == {
        "enabled": True,
        "height": 360,
        "width": 640,
        "resolution_source": "environment_builder_override",
    }
    assert run_metadata["rollout_limit"] == {
        "num_steps": 500,
        "num_episodes": None,
        "source": "configured",
    }
    assert run_metadata["num_rebuilds"] == 3

    recorder.finish("completed")
    completed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert completed_metadata["status"] == "completed"
    assert completed_metadata["finished_at"] is not None
    assert "error" not in completed_metadata


def test_metadata_marks_unconfigured_camera_dimensions_as_embodiment_defaults(tmp_path, monkeypatch):
    """Describe enabled production cameras without guessing dimensions absent from the Run config."""
    monkeypatch.setattr(metadata_module, "_resolve_source_revision", lambda: {})
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text("runs: {}\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    recorder = ArenaExperimentMetadataRecorder.start(
        output_directory,
        experiment_config_path,
        [],
        _runner_arguments(),
        command=["experiment_runner.py"],
    )

    recorder.record_resolved_experiment(_resolved_experiment(camera_height=None, camera_width=None))
    metadata = json.loads((output_directory / ARENA_EXPERIMENT_METADATA_FILENAME).read_text(encoding="utf-8"))

    assert metadata["runs"]["camera_baseline"]["camera"] == {
        "enabled": True,
        "height": None,
        "width": None,
        "resolution_source": "embodiment_default",
    }


def test_metadata_recorder_persists_failure_status(tmp_path, monkeypatch):
    """Record an ordinary Experiment failure before SimulationApp can terminate the process."""
    monkeypatch.setattr(metadata_module, "_resolve_source_revision", lambda: {})
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text("runs: {}\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    recorder = ArenaExperimentMetadataRecorder.start(
        output_directory,
        experiment_config_path,
        [],
        _runner_arguments(),
        command=["experiment_runner.py"],
    )

    recorder.finish("failed", RuntimeError("rollout failed"))
    metadata = json.loads((output_directory / ARENA_EXPERIMENT_METADATA_FILENAME).read_text(encoding="utf-8"))

    assert metadata["status"] == "failed"
    assert metadata["finished_at"] is not None
    assert metadata["error"] == {"type": "RuntimeError", "message": "rollout failed"}


def test_revision_collection_failure_never_fails_or_discards_invocation_metadata(tmp_path, monkeypatch):
    """Isolate best-effort revision failure while retaining the reproducibility fields."""

    def raise_revision_error():
        raise RuntimeError("revision unavailable")

    monkeypatch.setattr(metadata_module, "_resolve_source_revision", raise_revision_error)
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text("runs: {}\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    ArenaExperimentMetadataRecorder.start(
        output_directory,
        experiment_config_path,
        [],
        _runner_arguments(),
        command=["experiment_runner.py"],
    )
    metadata = json.loads((output_directory / ARENA_EXPERIMENT_METADATA_FILENAME).read_text(encoding="utf-8"))

    assert metadata["status"] == "starting"
    assert metadata["experiment_config"]["path"] == str(experiment_config_path.resolve())
    assert metadata["command"] == ["experiment_runner.py"]
    assert metadata["revision"] == {
        "metadata_error": {
            "type": "RuntimeError",
            "message": "revision unavailable",
        }
    }


def test_metadata_write_failure_never_fails_the_experiment(tmp_path, monkeypatch, capsys):
    """Keep lifecycle calls non-throwing when the artifact path cannot be written."""
    monkeypatch.setattr(metadata_module, "_resolve_source_revision", lambda: {})
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text("runs: {}\n", encoding="utf-8")
    output_path_that_is_not_a_directory = tmp_path / "not-a-directory"
    output_path_that_is_not_a_directory.write_text("occupied", encoding="utf-8")

    recorder = ArenaExperimentMetadataRecorder.start(
        output_path_that_is_not_a_directory,
        experiment_config_path,
        [],
        _runner_arguments(),
        command=["experiment_runner.py"],
    )
    recorder.record_resolved_experiment(_resolved_experiment())
    recorder.finish("failed", RuntimeError("rollout failed"))

    assert "Could not write Arena Experiment metadata" in capsys.readouterr().err


def test_metadata_serialization_failure_never_fails_the_experiment(tmp_path, monkeypatch, capsys):
    """Keep the Experiment runnable when unexpected metadata cannot be serialized."""
    monkeypatch.setattr(metadata_module, "_resolve_source_revision", lambda: {})

    def raise_serialization_error(_value):
        raise TypeError("cannot serialize metadata")

    monkeypatch.setattr(metadata_module, "_json_safe", raise_serialization_error)
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text("runs: {}\n", encoding="utf-8")
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    recorder = ArenaExperimentMetadataRecorder.start(
        output_directory,
        experiment_config_path,
        [],
        _runner_arguments(),
        command=["experiment_runner.py"],
    )
    recorder.record_resolved_experiment(_resolved_experiment())
    recorder.finish("completed")

    assert "cannot serialize metadata" in capsys.readouterr().err
