# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay fixed-topology deformables across independent native Lab processes."""

import gymnasium as gym
import h5py
from pathlib import Path
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def environment_registration_callback() -> list[str]:
    """Use zero actions and a short episode to exercise native export and nodal restoration."""
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tests.test_record_demos import _register_with_test_device

    def advance_device(device):
        assert device._steps <= 15, "Recording did not finish the short topology regression episode."

    def recording_complete(env):
        return env.episode_length_buf >= 5

    remaining_args = _register_with_test_device("droid_deformable_pick_and_place", advance_device)
    cfg = gym.spec("droid_deformable_pick_and_place").kwargs["env_cfg_entry_point"]
    cfg.terminations.success = TerminationTermCfg(func=recording_complete)
    # Keep this active for the child process: neither recording nor replay may regenerate the mesh.
    patch("pytetwild.tetrahedralize", side_effect=AssertionError("Unexpected runtime tetrahedralization")).start()
    return remaining_args


@pytest.mark.with_subprocess
@pytest.mark.parametrize("backend", ["physx", pytest.param("newton", marks=pytest.mark.with_newton)])
def test_deformable_record_and_fresh_process_replay(tmp_path, backend):
    """Export real nodal state, then restore and play it twice without invoking fTetWild."""
    import isaaclab

    tools = Path(isaaclab.__file__).resolve().parents[3] / "scripts/tools"
    dataset = tmp_path / f"{backend}.hdf5"
    common_args = [
        "--task",
        "droid_deformable_pick_and_place",
        "--pick_object",
        f"deformable_cube_{backend}",
        "--embodiment",
        "droid_differential_ik",
        "--visualizer",
        "none",
        "--device",
        "cuda:0",
        "--dataset_file",
        str(dataset),
        "--external_callback",
        "isaaclab_arena.tests.test_deformable_replay.environment_registration_callback",
    ]
    record = run_subprocess(
        [
            TestConstants.python_path,
            str(tools / "record_demos.py"),
            *common_args,
            "--teleop_device",
            "keyboard",
            "--cloudxr_env",
            "none",
            "--num_demos",
            "1",
            "--num_success_steps",
            "1",
            "--no-reset_sim_buffer_each_episode",
        ],
        timeout_sec=240,
        capture_output=True,
    )
    assert "Recording session completed with 1 successful demonstrations" in record.stdout, record.stdout
    with h5py.File(dataset) as data:
        episode = next(iter(data["data"].values()))
        nodal_state = episode["initial_state/deformable_object/pick_object/nodal_position"][:]
        assert nodal_state.shape[-2:] == ((69, 3) if backend == "newton" else (21, 3))
    for _ in range(2):
        replay = run_subprocess(
            [
                TestConstants.python_path,
                str(tools / "replay_demos.py"),
                *common_args,
                "--experience",
                "isaaclab.python.kit",
                "--validate_success_rate",
            ],
            timeout_sec=240,
            capture_output=True,
        )
        assert "Successfully replayed: 1/1" in replay.stdout, replay.stdout + replay.stderr
