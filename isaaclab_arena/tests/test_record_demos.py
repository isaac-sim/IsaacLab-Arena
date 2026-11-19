# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify native Lab recording preserves Arena task success, placement and variations."""

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
VARIATION_ASSET_NAME = "pick_up_object"
VARIATION_MASS = 0.25
VARIATION_STEPS = 5
VARIATION_METADATA_PREFIX = "RECORD_DEMOS_TEST_VARIATION "


def _register_with_test_device(task_name: str, advance_fn, initialize_fn=None) -> list[str]:
    """Register a zero-input device whose callbacks run against the environment bound at startup.

    Args:
        task_name: Name of the Arena task to configure after registration.
        advance_fn: Called with the device before each robot action; ``_steps`` is one-based.
        initialize_fn: Optional callback called with the device after its environment is bound.

    Returns:
        Arguments left for the native Lab recording script.
    """
    from isaaclab.devices import DeviceBase, DeviceCfg, DevicesCfg
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.registries import DeviceRegistry
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as register_arena_environment,
    )

    environment = {}

    def bind_environment(env, env_ids):
        environment["env"] = env

    class ScriptedTestDevice(DeviceBase):
        def __init__(self, cfg):
            super().__init__()
            self._env = environment["env"]
            self._callbacks = {}
            self.reset()
            if initialize_fn is not None:
                initialize_fn(self)

        def reset(self):
            self._steps = 0

        def add_callback(self, key, func):
            self._callbacks[key] = func

        def advance(self):
            self._steps += 1
            advance_fn(self)
            return torch.zeros(self._env.action_manager.total_action_dim, device=self._env.device)

    device_cfg = DeviceCfg(class_type=ScriptedTestDevice)
    # Replace hardware input before registration; GR1 joint control has no keyboard retargeter.
    with patch.object(DeviceRegistry, "get_teleop_device_cfg", return_value=device_cfg):
        remaining_args = register_arena_environment()
    cfg = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    cfg.teleop_devices = DevicesCfg(devices={"keyboard": device_cfg})
    cfg.events.bind_record_demos_test_environment = EventTermCfg(func=bind_environment, mode="startup")
    return remaining_args


def environment_registration_callback() -> list[str]:
    """Register the production environment with scripted door movement and zero robot input."""

    def initialize_device(device):
        from isaaclab_arena.assets.registries import AssetRegistry

        device._microwave = AssetRegistry().get_asset_by_name("microwave")()
        articulation = device._env.scene[device._microwave.name]
        joint_ids, _ = articulation.find_joints(device._microwave.openable_joint_name)
        joint_id = joint_ids[0]
        limits = articulation.data.joint_pos_limits.torch[0, joint_id].tolist()
        print(DOOR_METADATA_PREFIX + json.dumps({"joint_id": joint_id, "limits": limits}), flush=True)

    def advance_device(device):
        env = device._env
        steps = device._steps
        assert steps <= MAX_STEPS_PER_EPISODE, "Lab did not export after the microwave door opened"
        openness = 0.2 if steps <= CLOSED_STEPS else 1.0
        # Use the same state-setting API as test_open_door; leave task success and export untouched.
        device._microwave.rotate_revolute_joint(env, env_ids=None, percentage=openness)
        if steps == CLOSED_STEPS + 1:
            print("RECORD_DEMOS_TEST_DOOR_OPENED", flush=True)

    return _register_with_test_device("gr1_open_microwave", advance_device, initialize_device)


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


def variation_environment_registration_callback() -> list[str]:
    """Keep the maintained relation environment and replace only operator input and completion."""
    import warp as wp
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.utils.pose import Pose

    def recording_complete(env):
        return env.episode_length_buf >= VARIATION_STEPS

    def advance_device(device):
        env = device._env
        steps = device._steps
        assert steps <= MAX_STEPS_PER_EPISODE, "Lab did not finish the recording episode"
        if steps == 1:
            episode_index = env.get_episode_index(0)
            record = env.variation_recorder[f"{VARIATION_ASSET_NAME}.mass"]
            recorded_mass = record.sample_for_episode(0, episode_index)
            assert recorded_mass is not None, "Reset variation was not recorded for this episode"
            mass = wp.to_torch(env.scene[VARIATION_ASSET_NAME].data.body_mass)[0, 0].item()
            assert mass == pytest.approx(VARIATION_MASS)
            assert recorded_mass.item() == pytest.approx(mass)
            placement_record = env.variation_recorder.placement_record
            assert placement_record is not None, "Relation placement recorder was not attached"
            placement = placement_record.sample_for_episode(0, episode_index)
            assert placement is not None, "Reset placement was not recorded for this episode"
            # Compare object (O) poses in the environment frame (E) before stepping physics.
            recorded_T_E_O = Pose.from_dict(placement["poses"][VARIATION_ASSET_NAME]).to_tensor(env.device)
            T_E_O = env.arena_world.get_pose_e(VARIATION_ASSET_NAME)[0]
            torch.testing.assert_close(T_E_O[:3], recorded_T_E_O[:3], atol=2e-5, rtol=0)
            assert torch.abs(torch.dot(T_E_O[3:], recorded_T_E_O[3:])).item() > 1 - 1e-5
            print(VARIATION_METADATA_PREFIX + json.dumps({"episode": episode_index, "mass": mass}), flush=True)

    remaining_args = _register_with_test_device("pick_and_place_maple_table", advance_device)
    cfg = gym.spec("pick_and_place_maple_table").kwargs["env_cfg_entry_point"]
    # End short demos without a human policy; placement, reset variation and native export stay real.
    cfg.terminations.success = TerminationTermCfg(func=recording_complete)
    return remaining_args


@pytest.mark.with_subprocess
def test_record_demos_with_placement_and_variation(tmp_path):
    """Hydra enables reset-time sampling and native Lab exports both reset-separated demos."""
    import isaaclab

    record_script = Path(isaaclab.__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    dataset_path = tmp_path / "placement_variation.hdf5"
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
            "pick_and_place_maple_table",
            "--embodiment",
            "franka_ik",
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
            "isaaclab_arena.tests.test_record_demos.variation_environment_registration_callback",
            f"{VARIATION_ASSET_NAME}.mass.enabled=true",
            f"{VARIATION_ASSET_NAME}.mass.sampler_cfg.low=[{VARIATION_MASS}]",
            f"{VARIATION_ASSET_NAME}.mass.sampler_cfg.high=[{VARIATION_MASS}]",
        ],
        capture_output=True,
    )
    reset_samples = [
        json.loads(line.removeprefix(VARIATION_METADATA_PREFIX))
        for line in result.stdout.splitlines()
        if line.startswith(VARIATION_METADATA_PREFIX)
    ]
    assert [sample["episode"] for sample in reset_samples] == [0, 1]
    assert [sample["mass"] for sample in reset_samples] == pytest.approx([VARIATION_MASS] * NUM_DEMOS)
    with h5py.File(dataset_path, "r") as dataset:
        assert len(dataset["data"]) == NUM_DEMOS
        for episode in dataset["data"].values():
            assert episode.attrs["success"]
            assert len(episode["actions"]) == VARIATION_STEPS
            actions = episode["actions"][:]
            np.testing.assert_array_equal(actions, np.zeros_like(actions))
