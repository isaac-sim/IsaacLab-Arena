# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
NUM_ENVS = 2
# Tolerance for position comparison (env-relative); allow some slack for sim/spawn.
POSITION_ATOL = 0.05


def get_object_set_test_environment(num_envs: int, initial_pose):
    """Build and return (env, object_set) for object-set initial-pose tests.

    initial_pose: Either a single Pose or a dict[str, Pose] (object name -> Pose).
    The env is reset and stepped once so scene buffers reflect the reset event.
    """
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )

    object_set = RigidObjectSet(
        name="object_set",
        objects=[cracker_box, tomato_soup_can],
        random_choice=False,
    )
    object_set.set_initial_pose(initial_pose)

    scene = Scene(assets=[background, object_set, destination_location])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_set_initial_pose_test",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(object_set, destination_location, background),
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = num_envs
    args_cli.headless = HEADLESS
    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()
    # Step once so the scene buffers (root_pose_w) are updated from the sim after the reset event.
    with torch.inference_mode():
        env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    return env, object_set


def _test_object_set_pose_per_object(simulation_app) -> bool:
    """With set_initial_pose(dict), each object in the set should appear at its assigned pose per env."""
    from isaaclab_arena.utils.pose import Pose

    pose_cracker = Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    pose_tomato = Pose(position_xyz=(0.35, 0.25, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    # Object names match asset registry names (cracker_box, tomato_soup_can)
    initial_pose = {"cracker_box": pose_cracker, "tomato_soup_can": pose_tomato}
    env, object_set = get_object_set_test_environment(NUM_ENVS, initial_pose)

    try:
        # With random_choice=False, env 0 gets first object (cracker_box), env 1 gets second (tomato_soup_can).
        poses = object_set.get_object_pose(env, is_relative=True)
        assert poses.shape == (NUM_ENVS, 7), f"Expected (num_envs, 7), got {poses.shape}"

        expected_env0 = torch.tensor(pose_cracker.position_xyz, device=env.device)
        expected_env1 = torch.tensor(pose_tomato.position_xyz, device=env.device)
        assert torch.allclose(poses[0, :3], expected_env0, atol=POSITION_ATOL), (
            f"Env 0 (cracker_box): expected ~{expected_env0.tolist()}, got {poses[0, :3].tolist()}"
        )
        assert torch.allclose(poses[1, :3], expected_env1, atol=POSITION_ATOL), (
            f"Env 1 (tomato_soup_can): expected ~{expected_env1.tolist()}, got {poses[1, :3].tolist()}"
        )
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()


def _test_object_set_single_pose(simulation_app) -> bool:
    """With set_initial_pose(Pose), all objects in the set should end up at the same pose in every env."""
    from isaaclab_arena.utils.pose import Pose

    single_pose = Pose(position_xyz=(0.38, 0.12, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    env, object_set = get_object_set_test_environment(NUM_ENVS, single_pose)

    try:
        poses = object_set.get_object_pose(env, is_relative=True)
        assert poses.shape == (NUM_ENVS, 7), f"Expected (num_envs, 7), got {poses.shape}"

        expected = torch.tensor(single_pose.position_xyz, device=env.device)
        for i in range(NUM_ENVS):
            assert torch.allclose(poses[i, :3], expected, atol=POSITION_ATOL), (
                f"Env {i}: expected ~{expected.tolist()}, got {poses[i, :3].tolist()}"
            )
        # All envs should have the same position (single pose applied to all).
        assert torch.allclose(poses[:, :3], poses[0:1, :3].expand(NUM_ENVS, 3), atol=1e-5), (
            "All envs should have the same pose"
        )
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()


def test_object_set_pose_per_object():
    result = run_simulation_app_function(_test_object_set_pose_per_object, headless=HEADLESS)
    assert result, f"Test {_test_object_set_pose_per_object.__name__} failed"


def test_object_set_single_pose():
    result = run_simulation_app_function(_test_object_set_single_pose, headless=HEADLESS)
    assert result, f"Test {_test_object_set_single_pose.__name__} failed"


if __name__ == "__main__":
    test_object_set_pose_per_object()
    test_object_set_single_pose()
