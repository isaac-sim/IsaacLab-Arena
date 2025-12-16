# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 100
HEADLESS = True

# Test description.
# We simulate for 100 steps. In the first 50, the door is closed.
# In the second 50, the door is opened every step and the environment
# resets every step and the door is closed again.
# We set the max episode length small such that while the door is open,
# the environment resets every few (currently 5) steps.
# Therefore we get some episodes with movement and some without.
# See the calculations below for an exact calculation.
EXPECTED_MOVEMENT_RATE_EPS = 1e-6


def _test_revolute_joint_moved_rate(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.metrics import compute_metrics
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    microwave.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    scene = Scene(assets=[background, microwave])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env_cfg = env_builder.compose_manager_cfg()
    env_cfg.episode_length_s = 0.10
    env = env_builder.make_registered(env_cfg)
    env.reset()

    try:

        # Run some zero actions.
        for idx in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                if idx > NUM_STEPS / 2:
                    microwave.open(env, env_ids=None)
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")

        # Calculate the expected door moved rate.
        num_steps_per_episode = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
        print(f"Number of steps per episode: {num_steps_per_episode}")
        num_steps_door_closed = NUM_STEPS / 2
        num_episodes_no_movement = int(num_steps_door_closed / num_steps_per_episode)
        print(f"Number of episodes no movement: {num_episodes_no_movement}")
        num_episodes_with_movement = metrics["num_episodes"] - num_episodes_no_movement
        expected_movement_rate = num_episodes_with_movement / metrics["num_episodes"]
        print(f"Expected movement rate: {expected_movement_rate}")
        print(f"Measured movement rate: {metrics['revolute_joint_moved_rate']}")
        assert abs(metrics["revolute_joint_moved_rate"] - expected_movement_rate) < EXPECTED_MOVEMENT_RATE_EPS

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_revolute_joint_moved_rate_metric():
    result = run_simulation_app_function(
        _test_revolute_joint_moved_rate,
        headless=HEADLESS,
    )
    assert result, f"Test {test_revolute_joint_moved_rate_metric.__name__} failed"


if __name__ == "__main__":
    test_revolute_joint_moved_rate_metric()
