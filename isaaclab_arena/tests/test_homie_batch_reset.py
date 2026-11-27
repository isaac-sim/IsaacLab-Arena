# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_homie_batch_reset(simulation_app):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy import g1_homie_policy

    robot = SimpleNamespace(get_joint_group_indices=lambda group: range(29))
    with (
        patch.object(g1_homie_policy, "retrieve_file_path", return_value="unused"),
        patch.object(g1_homie_policy, "OnnxInferenceSession"),
    ):
        batch, uninterrupted, fresh = (
            g1_homie_policy.G1HomiePolicyV2(robot, "config/g1_homie_v2.yaml", "stand,walk", size) for size in (2, 1, 1)
        )

    def observe(policy):
        count = policy.num_envs
        policy.set_observation({
            "q": np.ones((count, 29)),
            "dq": np.zeros((count, 29)),
            "floating_base_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (count, 1)),
            "floating_base_vel": np.zeros((count, 6)),
        })

    for policy in (batch, uninterrupted):
        policy.set_goal({
            "navigate_cmd": [0.2, 0, 0],
            "base_height_command": np.full((policy.num_envs, 1), 0.68),
            "torso_orientation_rpy_cmd": [0.1, 0.2, 0.3],
        })
        policy.action[:] = 0.4
        for _ in range(3):
            observe(policy)

    batch.reset(torch.tensor([0]))
    for policy in (batch, uninterrupted, fresh):
        observe(policy)
    np.testing.assert_array_equal(batch.obs_buffer[0], fresh.obs_buffer[0])
    np.testing.assert_array_equal(batch.obs_buffer[1], uninterrupted.obs_buffer[0])
    torch.testing.assert_close(batch.gait_indices[0], fresh.gait_indices[0])
    torch.testing.assert_close(batch.gait_indices[1], uninterrupted.gait_indices[0])
    return True


def test_homie_batch_reset():
    assert run_function_with_persistent_simulation_app(_test_homie_batch_reset)
