# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify ArenaWorld live queries and environment-owned geometry caching."""

import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_sphere_environment(num_envs: int):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name("sphere")()
    arena_environment = IsaacLabArenaEnvironment(
        name="arena_world_test",
        scene=Scene(assets=[sphere]),
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env = ArenaEnvBuilder(
        arena_environment,
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    return env, sphere.name


def _test_arena_world(_simulation_app) -> bool:
    num_envs = 2
    env, sphere_name = _make_sphere_environment(num_envs)
    arena_world = env.unwrapped.arena_world
    try:
        env.reset()
        initial_pose_w = arena_world.get_pose_w(sphere_name).clone()
        assert initial_pose_w.shape == (num_envs, 7)
        assert arena_world.get_linear_velocity_w(sphere_name).shape == (num_envs, 3)

        local_aabb = arena_world.get_local_aabb(sphere_name)
        assert local_aabb.min_point.shape == (num_envs, 3)
        assert local_aabb.max_point.shape == (num_envs, 3)
        assert arena_world.get_local_aabb(sphere_name) is local_aabb

        moved_pose_w = initial_pose_w.clone()
        moved_pose_w[:, 0] += 0.25
        env.unwrapped.scene[sphere_name].write_root_pose_to_sim(moved_pose_w)
        torch.testing.assert_close(arena_world.get_pose_w(sphere_name), moved_pose_w)
        assert arena_world.get_local_aabb(sphere_name) is local_aabb

        env.reset()
        assert arena_world.get_local_aabb(sphere_name) is local_aabb
    finally:
        env.close()

    arena_world.close()
    try:
        arena_world.get_pose_w(sphere_name)
    except AssertionError as error:
        assert "ArenaWorld is closed" in str(error)
    else:
        raise AssertionError("ArenaWorld accepted a query after its environment closed.")
    return True


def test_arena_world():
    assert run_function_with_persistent_simulation_app(_test_arena_world)


if __name__ == "__main__":
    test_arena_world()
