# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation.reset_distribution import ResetDistributionRecorder


def _env():
    poses = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [5.0, 6.0, 7.0, 0.0, 0.0, 0.0, -1.0]])

    def get_state(*, is_relative):
        assert is_relative
        return {"rigid_object": {"box": {"root_pose": poses}}}

    return SimpleNamespace(
        scene=SimpleNamespace(get_state=get_state),
        active_episode_mask=torch.tensor([True, True]),
        get_episode_index=lambda env_id: 0,
        poses=poses,
    )


def test_local_samples_statistics_and_rebuild_identity(tmp_path):
    env = _env()
    recorder = ResetDistributionRecorder(tmp_path, "run")
    recorder.capture(env, torch.tensor([0, 1]), rebuild_index=0, seed=7)
    env.poses[:, :3] += 2
    recorder.capture(env, torch.tensor([0]), rebuild_index=1, seed=8)
    recorder.close()
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "reset_samples.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["state"]["rigid_object"]["box"]["root_pose"][:3] == [1, 2, 3]
    assert rows[-1]["seed"] == 8
    assert rows[-1]["rebuild_index"] == 1
    summary = json.loads((tmp_path / "reset_distribution.json").read_text())
    stats = summary["fields"]["rigid_object/box/root_position"]
    assert summary["sample_count"] == stats["count"] == 3
    assert stats["mean"] == [3, 4, 5]
    assert stats["min"] == [1, 2, 3]
    assert stats["max"] == [5, 6, 7]
    assert stats["std"] == pytest.approx([(8 / 3) ** 0.5] * 3)
    assert "rigid_object/box/root_pose" not in summary["fields"]


def test_inactive_cells_and_empty_reset_are_ignored(tmp_path):
    env = _env()
    env.active_episode_mask[1] = False
    recorder = ResetDistributionRecorder(tmp_path, "run")
    recorder.capture(env, torch.tensor([], dtype=torch.long), rebuild_index=0, seed=7)
    recorder.capture(env, torch.tensor([0, 1]), rebuild_index=0, seed=7)
    recorder.close()
    assert json.loads((tmp_path / "reset_distribution.json").read_text())["sample_count"] == 1


def test_duplicates_and_nonfinite_states_are_rejected(tmp_path):
    env = _env()
    recorder = ResetDistributionRecorder(tmp_path, "run")
    recorder.capture(env, torch.tensor([0]), rebuild_index=0, seed=7)
    with pytest.raises(ValueError, match="Duplicate"):
        recorder.capture(env, torch.tensor([0]), rebuild_index=0, seed=7)
    env.poses[1, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        recorder.capture(env, torch.tensor([1]), rebuild_index=0, seed=7)
    recorder.close()
    assert json.loads((tmp_path / "reset_distribution.json").read_text())["sample_count"] == 1


def test_existing_samples_are_never_overwritten(tmp_path):
    recorder = ResetDistributionRecorder(tmp_path, "run")
    with pytest.raises(FileExistsError):
        ResetDistributionRecorder(tmp_path, "run")
    recorder.close()
    with pytest.raises(FileExistsError):
        ResetDistributionRecorder(tmp_path, "run")


def test_cli_reset_measurement_is_opt_in():
    import argparse

    from isaaclab_arena.evaluation.experiment_runner_cli import add_experiment_runner_arguments

    parser = argparse.ArgumentParser()
    add_experiment_runner_arguments(parser)
    assert not parser.parse_args([]).measure_reset_distribution
    assert parser.parse_args(["--measure_reset_distribution"]).measure_reset_distribution


def test_pinned_environment_without_active_episode_mask(tmp_path):
    env = _env()
    del env.active_episode_mask
    recorder = ResetDistributionRecorder(tmp_path, "run")
    recorder.capture(env, torch.tensor([0, 1]), rebuild_index=0, seed=7)
    recorder.close()
    assert json.loads((tmp_path / "reset_distribution.json").read_text())["sample_count"] == 2
