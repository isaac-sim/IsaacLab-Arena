# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise gear success, episode reset, and HDF5 export through Lab's recorder CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

SOURCE_STEPS = 5
MAX_EPISODE_STEPS = 80
NUM_DEMOS = 2
TARGET_PREFIX = "GEAR_RECORDING_TEST_TARGET "


def environment_registration_callback() -> list[str]:
    """Use production registration and success with a scripted gear-placement device."""
    import gymnasium as gym
    import torch

    from isaaclab.devices import DeviceBase, DeviceCfg, DevicesCfg
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.registries import DeviceRegistry
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as register_arena_environment,
    )

    environment = {}

    def bind_environment(env, env_ids):
        environment["env"] = env

    class GearPlacementDevice(DeviceBase):
        """Seat the gear once per episode while holding the robot still."""

        def __init__(self, cfg):
            super().__init__()
            self._env = environment["env"]
            self._callbacks = {}
            self.reset()

        def reset(self):
            self._steps = 0

        def add_callback(self, key, func):
            self._callbacks[key] = func

        def advance(self):
            self._steps += 1
            assert self._steps <= MAX_EPISODE_STEPS, "Recorder failed to export the seated gear."
            if self._steps == SOURCE_STEPS + 1:
                target = self._env.arena_world.get_pose_w("medium_gear_target").clone()
                gear = self._env.scene["gear_insertion_medium_gear"]
                gear.write_root_pose_to_sim_index(root_pose=target)
                gear.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self._env.device))
                print(TARGET_PREFIX + json.dumps(target[0].tolist()), flush=True)
            return torch.zeros(self._env.action_manager.total_action_dim, device=self._env.device)

    device_cfg = DeviceCfg(class_type=GearPlacementDevice)
    with patch.object(DeviceRegistry, "get_teleop_device_cfg", return_value=device_cfg):
        remaining_args = register_arena_environment()
    cfg = gym.spec("gear_insertion").kwargs["env_cfg_entry_point"]
    cfg.teleop_devices = DevicesCfg(devices={"keyboard": device_cfg})
    cfg.events.bind_gear_recording_test = EventTermCfg(func=bind_environment, mode="startup")
    return remaining_args


@pytest.mark.with_newton
@pytest.mark.with_subprocess
def test_gear_insertion_recording_exports_two_successful_episodes(tmp_path):
    """Native recording must save two distinct successful episodes at the requested path."""
    import h5py
    import numpy as np

    import isaaclab

    script = Path(isaaclab.__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    dataset_path = tmp_path / "gear_insertion.hdf5"
    result = run_subprocess(
        [
            TestConstants.python_path,
            str(script),
            "--visualizer",
            "kit",
            "--task",
            "gear_insertion",
            "--external_callback",
            "isaaclab_arena.tests.test_gear_insertion_recording.environment_registration_callback",
            "--device",
            "cuda:0",
            "--teleop_device",
            "keyboard",
            "--disable_external_cameras",
            "--cloudxr_env",
            "none",
            # Use the supported recording mode that preserves Newton's simulation buffers.
            "--no-reset_sim_buffer_each_episode",
            "--seed",
            "42",
            "--placement_seed",
            "42",
            "--step_hz",
            "50",
            "--num_success_steps",
            "1",
            "--num_demos",
            str(NUM_DEMOS),
            "--dataset_file",
            str(dataset_path),
        ],
        timeout_sec=180,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    assert f"Recording session completed with {NUM_DEMOS} successful demonstrations" in output, output
    targets = []
    for line in result.stdout.splitlines():
        if line.startswith(TARGET_PREFIX):
            targets.append(json.loads(line.removeprefix(TARGET_PREFIX)))
    assert len(targets) == NUM_DEMOS, output
    assert not np.allclose(targets[0][:2], targets[1][:2]), "The second recording must use a new placement."
    assert dataset_path.is_file(), "Recorder reported success without writing the requested dataset."
    with h5py.File(dataset_path, "r") as dataset:
        assert len(dataset["data"]) == NUM_DEMOS
        for episode, target in zip(dataset["data"].values(), targets, strict=True):
            assert episode.attrs["success"]
            actions = episode["actions"][:]
            assert SOURCE_STEPS < len(actions) <= MAX_EPISODE_STEPS
            np.testing.assert_array_equal(actions, np.zeros_like(actions))
            gear_poses = episode["states/rigid_object/gear_insertion_medium_gear/root_pose"][:]
            assert np.isfinite(gear_poses).all()
            assert np.linalg.norm(gear_poses[0, :2] - target[:2]) > 0.1
            np.testing.assert_allclose(gear_poses[-1, :3], target[:3], rtol=0, atol=0.005)
