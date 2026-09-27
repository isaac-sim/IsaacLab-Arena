# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 100
HEADLESS = True

# Env 0 lifts and replaces its settled object in the drawer; env 1 drops its object outside.
# Check success against completed-episode counts because the two trajectories have different durations.
# Expect every object to move, allowing 5% for an unfinished episode with little movement.
EXPECTED_OBJECT_MOVED_RATE = 1.0
ALLOWABLE_OBJECT_MOVED_RATE_ERROR = 0.05


def _test_success_rate_metric(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.terms.events import set_object_pose_per_env
    from isaaclab_arena.tests.utils.pick_and_place import lift_settled_objects_once
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    scene = Scene(assets=[background, cracker_box, destination_location])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    NUM_ENVS = 2
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, env_kwargs = env_builder.compose_manager_cfg()

    # Replace the pose reset term:
    # - from: constant per env,
    # - to: per env pose
    pose_list = [
        # Success (in the drawer)
        Pose(position_xyz=(0.0, -0.5, 0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        # Fail (out of the drawer)
        Pose(position_xyz=(-0.5, -0.5, 0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
        func=set_object_pose_per_env,
        mode="reset",
        params={
            "pose_list": pose_list,
            "asset_cfg": SceneEntityCfg(cracker_box.name),
        },
    )

    env = env_builder.make_registered(env_cfg, env_kwargs)
    env.reset()

    try:

        base_env = env.unwrapped
        # Lift env 0 once per episode; env 1 keeps its drop-failure trajectory.
        lifted_envs = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        completed_episodes = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
        expected_success_by_env = torch.arange(base_env.num_envs, device=base_env.device) == 0
        previous_episode = None
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                current_episode = base_env.get_episode_index(0)
                if current_episode != previous_episode:
                    lifted_envs[0] = False
                    previous_episode = current_episode
                lift_settled_objects_once(base_env, cracker_box.name, lifted_envs)
                actions = torch.zeros(env.action_space.shape, device=base_env.device)
                _, _, terminated, truncated, _ = env.step(actions)
                ended_episodes = terminated | truncated
                observed_success = base_env.termination_manager.get_term("success")
                torch.testing.assert_close(
                    observed_success[ended_episodes],
                    expected_success_by_env[ended_episodes],
                )
                completed_episodes += ended_episodes.long()

        assert bool((completed_episodes > 0).all()), "Both environments must complete at least one episode."
        expected_success_rate = completed_episodes[0].item() / completed_episodes.sum().item()

        metrics: MetricsDataCollection = env.unwrapped.compute_metrics()
        print(f"Metrics: {metrics}")
        assert "success_rate" in metrics.metric_data_entries
        assert "object_moved_rate" in metrics.metric_data_entries
        success_rate = metrics.metric_data_entries["success_rate"].metric_value
        object_moved_rate = metrics.metric_data_entries["object_moved_rate"].metric_value
        print(f"Success rate: {success_rate}")
        print(f"Object moved rate: {object_moved_rate}")
        assert abs(success_rate - expected_success_rate) < 1e-6
        assert abs(object_moved_rate - EXPECTED_OBJECT_MOVED_RATE) < ALLOWABLE_OBJECT_MOVED_RATE_ERROR

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def test_success_rate_metric():
    result = run_function_with_persistent_simulation_app(
        _test_success_rate_metric,
        headless=HEADLESS,
    )
    assert result, f"Test {test_success_rate_metric.__name__} failed"


if __name__ == "__main__":
    test_success_rate_metric()
