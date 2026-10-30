# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the hold action that envs apply while they wait for a delayed policy action."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_ENVS = 2
HEADLESS = True


def _make_droid_env(embodiment: str):
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

    environment_cfg = PickAndPlaceMapleTableEnvironmentCfg(embodiment=embodiment)
    arena_environment = EnvironmentRegistry().get_factory_type_for_cfg(environment_cfg)().build(environment_cfg)
    return ArenaEnvBuilder(arena_environment, ArenaEnvBuilderCfg(num_envs=NUM_ENVS)).make_registered()


def _test_hold_action_keeps_the_absolute_joint_position_arm_in_place(simulation_app):
    """Absolute joint position terms hold the current joint positions; the gripper repeats its last command."""
    import torch

    from isaaclab_arena.policy.inference_latency import hold_action_from_action_manager

    env = _make_droid_env("droid_abs_joint_pos")
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        arm_joint_ids = env.unwrapped.action_manager.get_term("arm_action")._joint_ids
        joint_pos_before = robot.data.joint_pos.torch[:, arm_joint_ids].clone()

        hold_action = hold_action_from_action_manager(env)
        assert hold_action.shape == (NUM_ENVS, env.unwrapped.action_manager.total_action_dim)
        torch.testing.assert_close(hold_action[:, :7], joint_pos_before)
        assert torch.all(hold_action[:, 7] == 0.0), "the gripper repeats its last command, zero after a reset"

        for _ in range(5):
            env.step(hold_action_from_action_manager(env))
        joint_pos_after = robot.data.joint_pos.torch[:, arm_joint_ids]
        assert torch.allclose(
            joint_pos_after, joint_pos_before, atol=2e-2
        ), f"arm drifted while holding: {(joint_pos_after - joint_pos_before).abs().max().item():.4f} rad"
    finally:
        env.close()
    return True


def _test_hold_action_is_zero_for_relative_joint_position_terms(simulation_app):
    """Relative joint position terms hold with a zero delta."""
    import torch

    from isaaclab_arena.policy.inference_latency import hold_action_from_action_manager

    env = _make_droid_env("droid_rel_joint_pos")
    try:
        env.reset()
        hold_action = hold_action_from_action_manager(env)
        assert hold_action.shape == (NUM_ENVS, env.unwrapped.action_manager.total_action_dim)
        assert torch.all(hold_action == 0.0)
    finally:
        env.close()
    return True


@pytest.mark.parametrize(
    "test_function",
    [
        _test_hold_action_keeps_the_absolute_joint_position_arm_in_place,
        _test_hold_action_is_zero_for_relative_joint_position_terms,
    ],
    ids=lambda function: function.__name__.removeprefix("_test_"),
)
def test_hold_action(test_function):
    assert run_function_with_persistent_simulation_app(test_function, headless=HEADLESS)
