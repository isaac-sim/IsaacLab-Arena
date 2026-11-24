# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for gear insertion built as an Isaac Lab MDP2 environment from Arena's environment, on Newton."""

import math

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_ENVS = 2
BUDGET = 3
GEAR_NAME = "gear_insertion_medium_gear"
BASE_NAME = "gear_insertion_base"
CAMERA_NAMES = ("external_camera", "external_camera_2", "wrist_camera")


def _make_mdp2_gear_env(episode_budget: int | None = None):
    """Build the registered gear insertion environment with DROID joint position actions as an MDP2 environment."""
    import argparse

    from isaaclab_experimental.envs import MDP2RLEnv

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.mdp2.env_builder import ArenaMDP2EnvBuilder
    from isaaclab_arena_environments.cli import (
        add_environment_cli_args,
        build_environment_from_cli,
        ensure_environments_registered,
    )

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name("gear_insertion")
    parser = argparse.ArgumentParser()
    add_environment_cli_args(parser, factory_type)
    args = parser.parse_args(["--embodiment", "droid_abs_joint_pos", "--camera_width", "320", "--camera_height", "180"])
    args.enable_cameras = True
    args.hdr = None
    arena = build_environment_from_cli(factory_type, args)
    # MDP2 captures the physics in CUDA graphs, which Isaac RTX cameras cannot take part in.
    builder_cfg = ArenaEnvBuilderCfg(num_envs=NUM_ENVS, seed=42, placement_seed=42, camera_renderer="newton_warp")
    cfg = ArenaMDP2EnvBuilder(arena, builder_cfg).build_mdp2_env_cfg()
    cfg.sim.device = "cuda:0"
    cfg.episode_budget = episode_budget
    return MDP2RLEnv(cfg), arena, args.position_randomization_m


def _test_mdp2_gear_insertion_episodes(simulation_app) -> bool:
    import torch

    from isaaclab.utils.math import quat_apply

    from isaaclab_arena_environments.gear_insertion_environment import GEAR_BASE_XY, GEAR_INITIAL_XY

    env, arena, offset = _make_mdp2_gear_env(episode_budget=BUDGET)
    try:
        obs, _ = env.reset()
        policy, cameras = obs["policy"], obs["camera_obs"]
        assert {name: tuple(value.shape[1:]) for name, value in policy.items()} == {
            "actions": (8,),
            "robot_joint_pos": (13,),
            "joint_pos": (7,),
            "gripper_pos": (1,),
            "eef_pos": (3,),
            "eef_quat": (4,),
        }
        for name in CAMERA_NAMES:
            image = cameras[f"{name}_rgb"]
            assert image.shape == (NUM_ENVS, 180, 320, 3) and image.dtype == torch.uint8
            assert image.float().std() > 10.0  # Rendered, not blank

        # The relation-placement event put both parts within their position limits.
        robot, base, gear = env.scene["robot"], env.scene[BASE_NAME], env.scene[GEAR_NAME]
        origins = env.scene.env_origins
        for part, nominal_xy in ((base, GEAR_BASE_XY), (gear, GEAR_INITIAL_XY)):
            t_E_P = part.data.root_link_pose_w.torch[:, :3] - origins
            assert ((t_E_P[:, :2] - torch.tensor(nominal_xy, device=env.device)).abs() <= offset + 1e-4).all()

        # Hold the arm at its approach pose and close the gripper.
        actions = torch.zeros(NUM_ENVS, 8, device=env.device)
        actions[:, :7] = robot.data.default_joint_pos.torch[:, :7]
        actions[:, 7] = 1.0
        env.step(actions)
        _, _, terminated, truncated, _ = env.step(actions)
        assert not (terminated | truncated).any()
        assert env.is_captured and env.num_graphs == 1  # The whole step, cameras included, replays as one graph

        # Seat the gear of world 0 on its peg; the gear of world 1 stays on the table.
        t_B_T = torch.tensor(
            arena.task.insertion_target.initial_pose_relative_to_parent.position_xyz, device=env.device
        )
        T_W_B = base.data.root_link_pose_w.torch
        pose = gear.data.root_link_pose_w.torch.clone()
        pose[0, :3] = T_W_B[0, :3] + quat_apply(T_W_B[0:1, 3:7], t_B_T[None])[0]
        pose[0, 3:7] = T_W_B[0, 3:7]
        gear.write_root_pose_to_sim_index(root_pose=pose)
        gear.write_root_velocity_to_sim_index(root_velocity=torch.zeros(NUM_ENVS, 6, device=env.device))
        required = arena.task.success_criteria.consecutive_success_steps
        terminated_steps = []
        for _ in range(2 * required):
            _, _, terminated, _, _ = env.step(actions)
            terminated_steps.append(terminated.tolist())
            if terminated[0]:
                break
        assert terminated_steps == [[False, False]] * (required - 1) + [[True, False]], terminated_steps
        records = env.episode_records()
        assert records["world"].tolist() == [0.0]
        # The task was done at the last step of the episode, which it ended as a success.
        assert records["progress/task"].tolist() == records["length"].tolist()
        assert records["termination/success"].tolist() == [1.0]

        # Drop both gears below the background's minimum height, beside the table and clear of the robot's stand:
        # the budget of three is then spent.
        def drop() -> None:
            pose = gear.data.root_link_pose_w.torch.clone()
            pose[:, :3] = origins + torch.tensor([0.5, -0.7, -0.3], device=env.device)
            gear.write_root_pose_to_sim_index(root_pose=pose)
            gear.write_root_velocity_to_sim_index(root_velocity=torch.zeros(NUM_ENVS, 6, device=env.device))

        drop()
        _, _, terminated, _, _ = env.step(actions)
        assert terminated.tolist() == [True, True] and env.episodes_completed == BUDGET
        records = env.episode_records()
        assert records["world"].tolist() == [0.0, 0.0, 1.0]
        assert records["termination/gear_dropped"].tolist() == [0.0, 1.0, 1.0]
        assert records["progress/task"].tolist()[1:] == [-1.0, -1.0]  # Dropped, never done
        drop()
        _, _, terminated, truncated, _ = env.step(actions)
        assert not (terminated | truncated).any() and env.episodes_completed == BUDGET
    finally:
        env.close()
    return True


def _test_mdp2_gear_insertion_seated_thresholds(simulation_app) -> bool:
    import torch

    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

    from isaaclab_arena.mdp2 import task_terms

    thresholds = {
        "target_offset": (0.0366185, 0.0, 0.0075),
        "max_lateral": 0.0035,
        "max_depth_below": 0.002,
        "max_height_above": 0.001,
        "max_tilt_rad": math.radians(5.0),
        "max_linear_speed": 0.05,
        "max_angular_speed": 0.5,
    }
    q_W_B = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2]))
    T_W_B = torch.cat([torch.tensor([[0.5, -0.1, 0.09]]), q_W_B], dim=-1)
    t_W_T = T_W_B[:, :3] + quat_apply(q_W_B, torch.tensor([thresholds["target_offset"]]))

    def seated(t_T_S=(0.0, 0.0, 0.0), tilt_deg=0.0, linear_speed=0.0, angular_speed=0.0) -> float:
        tilt = quat_from_euler_xyz(torch.tensor([math.radians(tilt_deg)]), torch.zeros(1), torch.zeros(1))
        T_W_S = torch.cat([t_W_T + quat_apply(q_W_B, torch.tensor([t_T_S])), quat_mul(q_W_B, tilt)], dim=-1)
        velocity = torch.tensor([[linear_speed, 0.0, 0.0, 0.0, 0.0, angular_speed]])
        return task_terms._seated_at_target(T_W_S, velocity, T_W_B, **thresholds).item()

    assert seated() == 1.0
    assert seated(t_T_S=(0.0034, 0.0, 0.0)) == 1.0 and seated(t_T_S=(0.0, 0.0036, 0.0)) == 0.0
    assert seated(t_T_S=(0.0, 0.0, -0.0019)) == 1.0 and seated(t_T_S=(0.0, 0.0, -0.0021)) == 0.0
    assert seated(t_T_S=(0.0, 0.0, 0.0009)) == 1.0 and seated(t_T_S=(0.0, 0.0, 0.0011)) == 0.0
    assert seated(tilt_deg=4.0) == 1.0 and seated(tilt_deg=6.0) == 0.0
    assert seated(linear_speed=0.04) == 1.0 and seated(linear_speed=0.06) == 0.0
    assert seated(angular_speed=0.4) == 1.0 and seated(angular_speed=0.6) == 0.0
    return True


@pytest.mark.with_newton
@pytest.mark.with_cameras
def test_mdp2_gear_insertion_episodes():
    assert run_function_with_persistent_simulation_app(_test_mdp2_gear_insertion_episodes, enable_cameras=True)


def test_mdp2_gear_insertion_seated_thresholds():
    assert run_function_with_persistent_simulation_app(_test_mdp2_gear_insertion_seated_thresholds)
