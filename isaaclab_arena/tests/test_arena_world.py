# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify ArenaWorld live queries and environment-owned geometry caching."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_sphere_environment(num_envs: int):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name("sphere")()
    arena_environment = IsaacLabArenaEnvironment(
        name="arena_world_test",
        scene=Scene(assets=[sphere]),
        embodiment=FrankaJointPosEmbodiment(),
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env = ArenaEnvBuilder(
        arena_environment,
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    return env, sphere.name


def _test_arena_world(_simulation_app) -> bool:
    import torch

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.utils.joint_utils import get_unnormalized_joint_position

    num_envs = 2
    env, sphere_name = _make_sphere_environment(num_envs)
    arena_world = env.unwrapped.arena_world
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        joint_name = "panda_finger_joint1"
        joint_index = robot.data.joint_names.index(joint_name)
        measured = arena_world.get_joint_position("robot", joint_name)
        torch.testing.assert_close(measured, robot.data.joint_pos.torch[:, joint_index])
        torch.testing.assert_close(
            measured, get_unnormalized_joint_position(env, SceneEntityCfg("robot", joint_names=[joint_name]))
        )
        body_index = robot.data.body_names.index("panda_hand")
        T_W_B = arena_world.get_body_pose_w("robot", "panda_hand")
        torch.testing.assert_close(T_W_B, robot.data.body_link_pose_w.torch[:, body_index])
        # Isaac Lab 3.0 uses XYZW quaternions; verify the configured offset against a live frame transformer.
        offset_B = T_W_B.new_tensor(env.unwrapped.cfg.scene.ee_frame.target_frames[0].offset.pos)
        expected_ee_position_W = T_W_B[:, :3] + quat_apply(T_W_B[:, 3:], offset_B.expand(num_envs, -1))
        torch.testing.assert_close(
            arena_world.get_frame_position_w("ee_frame", "end_effector"), expected_ee_position_W, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            arena_world.get_frame_position_w("ee_frame"), expected_ee_position_W, atol=1e-5, rtol=1e-5
        )
        # S is the sphere frame.
        T_W_S_initial = arena_world.get_pose_w(sphere_name).clone()
        assert T_W_S_initial.shape == (num_envs, 7)
        assert arena_world.get_root_linear_velocity_w(sphere_name).shape == (num_envs, 3)
        assert arena_world.get_root_angular_velocity_w(sphere_name).shape == (num_envs, 3)

        sphere_bounds_S = arena_world.get_aabb_in_local_frame(sphere_name)
        assert sphere_bounds_S.min_point.shape == (num_envs, 3)
        assert sphere_bounds_S.max_point.shape == (num_envs, 3)
        assert arena_world.get_aabb_in_local_frame(sphere_name) is sphere_bounds_S

        T_W_S_moved = T_W_S_initial.clone()
        T_W_S_moved[:, 0] += 0.25
        env.unwrapped.scene[sphere_name].write_root_pose_to_sim(T_W_S_moved)
        torch.testing.assert_close(arena_world.get_pose_w(sphere_name), T_W_S_moved)
        assert arena_world.get_aabb_in_local_frame(sphere_name) is sphere_bounds_S

        env.reset()
        assert arena_world.get_aabb_in_local_frame(sphere_name) is sphere_bounds_S
    finally:
        env.close()
    return True


def test_arena_world():
    assert run_function_with_persistent_simulation_app(_test_arena_world)


def _test_arena_world_articulation_queries(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.environments.arena_world import ArenaWorld

    joint_positions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    body_poses = torch.tensor(
        [[[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]], [[0, 0, 0, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]],
        dtype=torch.float32,
    )
    data = SimpleNamespace(
        joint_names=["unused", "finger"],
        joint_pos=SimpleNamespace(torch=joint_positions),
        body_names=["base", "wrist"],
        body_link_pose_w=SimpleNamespace(torch=body_poses),
    )
    scene = SimpleNamespace(num_envs=2, articulations={"robot": SimpleNamespace(data=data)})
    world = ArenaWorld(scene)
    torch.testing.assert_close(world.get_joint_position("robot", "finger"), joint_positions[:, 1])
    torch.testing.assert_close(world.get_body_pose_w("robot", "wrist"), body_poses[:, 1])

    # Replacing the backing tensors must not leave cached state behind.
    data.joint_pos.torch = joint_positions + 0.5
    data.body_link_pose_w.torch = body_poses.clone()
    data.body_link_pose_w.torch[:, 1, 0] += 0.5
    torch.testing.assert_close(world.get_joint_position("robot", "finger"), joint_positions[:, 1] + 0.5)
    torch.testing.assert_close(world.get_body_pose_w("robot", "wrist"), data.body_link_pose_w.torch[:, 1])

    with pytest.raises(AssertionError, match="must name an articulation"):
        world.get_joint_position("missing", "finger")
    with pytest.raises(AssertionError, match="has no joint"):
        world.get_joint_position("robot", "missing")
    with pytest.raises(AssertionError, match="has no body"):
        world.get_body_pose_w("robot", "missing")
    return True


def test_arena_world_articulation_queries() -> None:
    assert run_function_with_persistent_simulation_app(_test_arena_world_articulation_queries)


if __name__ == "__main__":
    test_arena_world()
    test_arena_world_articulation_queries()
