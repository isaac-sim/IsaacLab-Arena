# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check reset camera frames through Lab's native deformable recording workflow."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def environment_registration_callback() -> list[str]:
    """Record short stationary episodes using the production deformable scene and cameras."""
    import gymnasium as gym

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tests.test_record_demos import _register_with_test_device

    def advance_device(device):
        assert device._steps <= 10, "Recorder did not export the stationary episode."

    remaining_args = _register_with_test_device("droid_deformable_pick_and_place", advance_device)
    cfg = gym.spec("droid_deformable_pick_and_place").kwargs["env_cfg_entry_point"]
    # This test checks camera recording, independently of manipulating the deformable object.
    cfg.terminations.success = TerminationTermCfg(func=lambda env: env.episode_length_buf >= 3)
    return remaining_args


@pytest.mark.with_subprocess
@pytest.mark.parametrize("visualizer", ["none", "kit"])
@pytest.mark.parametrize("physics_backend", [pytest.param("newton", marks=pytest.mark.with_newton), "physx"])
def test_deformable_recording_first_camera_frames(tmp_path, physics_backend, visualizer):
    """All cameras must record a populated first frame after initial and subsequent resets."""
    import h5py
    import numpy as np

    import isaaclab

    script = Path(isaaclab.__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    dataset_path = tmp_path / "deformable.hdf5"
    result = run_subprocess(
        [
            TestConstants.python_path,
            str(script),
            "--visualizer",
            visualizer,
            "--task",
            "droid_deformable_pick_and_place",
            "--pick_object",
            f"deformable_cube_{physics_backend}",
            "--external_callback",
            "isaaclab_arena.tests.test_deformable_recording.environment_registration_callback",
            "--device",
            "cuda:0",
            "--teleop_device",
            "keyboard",
            "--enable_cameras",
            "--cloudxr_env",
            "none",
            "--seed",
            "42",
            "--placement_seed",
            "42",
            "--step_hz",
            "50",
            "--num_success_steps",
            "1",
            "--num_demos",
            "2",
            "--dataset_file",
            str(dataset_path),
        ],
        capture_output=True,
    )
    assert "Recording session completed with 2 successful demonstrations" in result.stdout, (
        result.stdout + result.stderr
    )
    with h5py.File(dataset_path, "r") as dataset:
        assert len(dataset["data"]) == 2
        for episode_name, episode in dataset["data"].items():
            assert len(episode["camera_obs"]) == 3
            for camera_name, images in episode["camera_obs"].items():
                first_frame = images[0]
                assert first_frame.shape == (720, 1280, 3)
                assert first_frame.dtype == np.uint8
                assert (
                    float(first_frame.mean()) > 1.0
                ), f"{episode_name}/{camera_name}: first frame is black (mean={first_frame.mean():.1f})."
                assert first_frame.max() > first_frame.min(), f"{episode_name}/{camera_name}: first frame is blank."
