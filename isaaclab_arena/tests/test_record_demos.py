# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify native Lab recording exports GR1 microwave demos through Arena task success."""

from __future__ import annotations

import gymnasium as gym
import h5py
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

CLOSED_STEPS = 5
MAX_STEPS_PER_EPISODE = 30
NUM_DEMOS = 2
DOOR_METADATA_PREFIX = "RECORD_DEMOS_TEST_DOOR "


def environment_registration_callback() -> list[str]:
    """Register the production environment with scripted door movement and zero robot input."""
    from isaaclab.devices import DeviceBase, DeviceCfg, DevicesCfg
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as register_arena_environment,
    )

    environment = {}

    def bind_environment(env, env_ids):
        environment["env"] = env

    class MicrowaveTestDevice(DeviceBase):
        """Hold the robot still and open the real microwave joint after a closed prefix."""

        def __init__(self, cfg):
            super().__init__()
            self._env = environment["env"]
            self._microwave = AssetRegistry().get_asset_by_name("microwave")()
            self._callbacks = {}
            self.reset()
            articulation = self._env.scene[self._microwave.name]
            joint_ids, _ = articulation.find_joints(self._microwave.openable_joint_name)
            joint_id = joint_ids[0]
            limits = articulation.data.joint_pos_limits.torch[0, joint_id].tolist()
            print(DOOR_METADATA_PREFIX + json.dumps({"joint_id": joint_id, "limits": limits}), flush=True)

        def reset(self):
            self._steps = 0

        def add_callback(self, key, func):
            self._callbacks[key] = func

        def advance(self):
            self._steps += 1
            assert self._steps <= MAX_STEPS_PER_EPISODE, "Lab did not export after the microwave door opened"
            openness = 0.2 if self._steps <= CLOSED_STEPS else 1.0
            # Use the same state-setting API as test_open_door; leave task success and export untouched.
            self._microwave.rotate_revolute_joint(self._env, env_ids=None, percentage=openness)
            if self._steps == CLOSED_STEPS + 1:
                print("RECORD_DEMOS_TEST_DOOR_OPENED", flush=True)
            return torch.zeros(self._env.action_manager.total_action_dim, device=self._env.device)

    device_cfg = DeviceCfg(class_type=MicrowaveTestDevice)
    # GR1 joint control has no keyboard retargeter. Replace only the hardware input configuration.
    with patch.object(DeviceRegistry, "get_teleop_device_cfg", return_value=device_cfg):
        remaining_args = register_arena_environment()
    cfg = gym.spec("gr1_open_microwave").kwargs["env_cfg_entry_point"]
    cfg.teleop_devices = DevicesCfg(devices={"keyboard": device_cfg})
    cfg.events.bind_record_demos_test_environment = EventTermCfg(func=bind_environment, mode="startup")
    return remaining_args


@pytest.mark.with_subprocess
def test_record_demos_gr1_open_microwave(tmp_path):
    """Lab must recognize native task success and automatically export two reset-separated demos."""
    import isaaclab

    # Resolve Lab independently of the Arena branch so this same file can run in a main worktree.
    lab_root = Path(isaaclab.__file__).resolve().parents[3]
    record_script = lab_root / "scripts/tools/record_demos.py"
    assert record_script.is_file(), f"Native Lab recording script is missing: {record_script}"
    dataset_path = tmp_path / "gr1_open_microwave.hdf5"
    result = run_subprocess(
        [
            TestConstants.python_path,
            str(record_script),
            "--visualizer",
            "kit",
            "--device",
            "cpu",
            "--disable_external_cameras",
            "--task",
            "gr1_open_microwave",
            "--embodiment",
            "gr1_joint",
            "--teleop_device",
            "keyboard",
            "--step_hz",
            "50",
            "--num_success_steps",
            "1",
            "--num_demos",
            str(NUM_DEMOS),
            "--dataset_file",
            str(dataset_path),
            "--external_callback",
            "isaaclab_arena.tests.test_record_demos.environment_registration_callback",
        ],
        timeout_sec=180,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert f"Recording session completed with {NUM_DEMOS} successful demonstrations" in output, output
    assert result.stdout.count("RECORD_DEMOS_TEST_DOOR_OPENED") == NUM_DEMOS, output
    metadata_line = next(line for line in result.stdout.splitlines() if line.startswith(DOOR_METADATA_PREFIX))
    door_metadata = json.loads(metadata_line.removeprefix(DOOR_METADATA_PREFIX))
    joint_id = door_metadata["joint_id"]
    lower, upper = door_metadata["limits"]

    assert dataset_path.is_file(), "Lab reported success without saving a dataset"
    with h5py.File(dataset_path, "r") as dataset:
        assert len(dataset["data"]) == NUM_DEMOS, "Expected both successful episodes to be exported"
        for episode in dataset["data"].values():
            assert episode.attrs["success"], "The exported episode was not marked successful"
            actions = episode["actions"][:]
            assert CLOSED_STEPS + 2 <= len(actions) <= MAX_STEPS_PER_EPISODE
            assert np.isfinite(actions).all()
            np.testing.assert_array_equal(actions, np.zeros_like(actions))
            joint_positions = episode["states/articulation/microwave/joint_position"][:, joint_id]
            openness = (joint_positions - lower) / (upper - lower)
            # Openable reverses the normalized direction for joints with a negative lower limit.
            if lower < 0.0:
                openness = 1.0 - openness
            np.testing.assert_allclose(openness[:CLOSED_STEPS], 0.2, atol=0.02, rtol=0.0)
            assert openness[-1] >= 0.8, "The saved demo does not end with an open microwave door"
