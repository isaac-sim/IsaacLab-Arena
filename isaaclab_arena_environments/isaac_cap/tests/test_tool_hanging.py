# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check the tool-hanging goal geometry and success in the easy wrench environment.

The simulation test teleports the wrench so its ring sits over the hook's shank and lets it
settle under gravity. It verifies that the task reports success and resets the episode.
"""

import torch
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_environments.isaac_cap.tool_hanging.geometry import goal_geometry_from_dict

pytestmark = pytest.mark.isaac_cap

# Tool at the origin; fixture translated by one meter along X with a quarter turn about Z.
T_W_T = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
T_W_X = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 2**-0.5, 2**-0.5]])


def test_loop_on_rod_uses_closest_segment_point():
    # The fixture's local X rod becomes a world Y segment through (1, 0, 0).
    goal = goal_geometry_from_dict({
        "loop": {"center_xyz": [1.0, 0.3, 0.005], "radius_m": 0.01},
        "rod": {"start_xyz": [0, 0, 0], "end_xyz": [0.5, 0, 0]},
    })
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [True]
    goal.loops[0].center_xyz = (1.0, 0.6, 0.0)  # beyond the rod end
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [False]


def test_point_in_box_ignores_fixture_rotation():
    goal = goal_geometry_from_dict({
        "containment": {
            "minimum_xyz": [-1.1, -0.1, -0.1],
            "maximum_xyz": [-0.9, 0.1, 0.1],
            "point_xyz": [0.0, 0.0, 0.05],
        }
    })
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [True]
    goal.point_xyz = (0.0, 0.0, 0.2)
    assert goal.evaluate(T_W_T, T_W_X).tolist() == [False]


def _test_wrench_hangs_on_hook(_simulation_app):
    from isaaclab.utils.math import quat_apply, quat_from_matrix

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena_environments.isaac_cap import tool_hanging

    config_dir = Path(tool_hanging.__file__).parent
    experiment = load_arena_experiment_from_config_file(
        config_dir / "experiment_configs/tool_hanging_zero_action_experiment.yaml", device="cuda:0"
    )
    task_params = ArenaEnvGraphSpec.from_yaml(config_dir / "wrench_easy.yaml").task.subtasks[0].params
    goal = task_params["goals"][0]
    env = build_arena_builder_from_run_cfg(experiment.runs["tool_hanging"]).make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        wrench = base.scene[task_params["tools"][0]]
        # W is world, H is the hook, and T is the wrench root frame. Take the goal's rod midpoint
        # in H and its ring center in T from the scene YAML.
        T_W_H = base.arena_world.get_pose_w(task_params["fixtures"][0])
        rod = goal["rod"]
        shank_center_H = 0.5 * (T_W_H.new_tensor([rod["start_xyz"]]) + T_W_H.new_tensor([rod["end_xyz"]]))
        shank_center_W = T_W_H[:, :3] + quat_apply(T_W_H[:, 3:], shank_center_H)
        # Hang the wrench vertically: its ring normal (local +Z) follows the shank (world X) and
        # its handle (local +X from the ring) points down, so the ring drops onto the shank.
        R_W_T = T_W_H.new_tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]])
        q_W_T = quat_from_matrix(R_W_T)
        ring_center_T = T_W_H.new_tensor([goal["loop"]["center_xyz"]])
        T_W_T = torch.cat((shank_center_W - quat_apply(q_W_T, ring_center_T), q_W_T), dim=-1)
        wrench.write_root_pose_to_sim(T_W_T)
        wrench.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        with torch.inference_mode():
            for step in range(300):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                success = base.termination_manager.get_term("success")
                if step == 0:
                    assert not success.any(), "A wrench that has not settled must not count as hung"
                assert not truncated.any(), "Hang test timed out"
                if terminated.any():
                    assert success.all(), "Episode ended without tool-hanging success"
                    assert base.episode_length_buf[0] == 0, "Success did not reset the environment"
                    return True
        assert False, "Wrench did not settle on the hook"
    finally:
        env.close()


@pytest.mark.with_newton
def test_wrench_hangs_on_hook():
    assert run_function_with_persistent_simulation_app(_test_wrench_hangs_on_hook)
