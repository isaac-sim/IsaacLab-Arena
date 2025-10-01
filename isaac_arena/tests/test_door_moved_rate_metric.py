# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 100
HEADLESS = False

# Test description.
# We start 2 envs. In these two envs:
# - 1 : The object falls in the drawer resulting in a success.
# - 2 : The object falls out of the drawer resulting in a failure.
# We expect the success rate to be 0.5 and the object moved rate to be 1.0.
# We allow for
# - Success rate error: 10% because the two environments reset at different rates
#   due to the different height that the object falls from.
# - Object moved rate error: 5% to allow for the case where in the last run
#   the object doesn't move much (in practice I haven't seen this happen).
EXPECTED_SUCCESS_RATE = 0.5
ALLOWABLE_SUCCESS_RATE_ERROR = 0.1
EXPECTED_OBJECT_MOVED_RATE = 1.0
ALLOWABLE_OBJECT_MOVED_RATE_ERROR = 0.05


def _test_door_moved_rate(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.metrics.metrics import compute_metrics
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.open_door_task import OpenDoorTask
    from isaac_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    microwave.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    scene = Scene(assets=[background, microwave])
    isaac_arena_environment = IsaacArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    args_cli = get_isaac_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env_cfg = env_builder.compose_manager_cfg()
    env_cfg.episode_length_s = 0.10
    # env_cfg.terminations.time_out.time_out = True
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
        # assert "success_rate" in metrics
        # assert "object_moved_rate" in metrics
        # success_rate = metrics["success_rate"]
        # object_moved_rate = metrics["object_moved_rate"]
        # print(f"Success rate: {success_rate}")
        # print(f"Object moved rate: {object_moved_rate}")
        # assert abs(success_rate - EXPECTED_SUCCESS_RATE) < ALLOWABLE_SUCCESS_RATE_ERROR
        # assert abs(object_moved_rate - EXPECTED_OBJECT_MOVED_RATE) < ALLOWABLE_OBJECT_MOVED_RATE_ERROR

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_door_moved_rate_metric():
    result = run_simulation_app_function_in_separate_process(
        _test_door_moved_rate,
        headless=HEADLESS,
    )
    assert result, f"Test {test_door_moved_rate_metric.__name__} failed"


if __name__ == "__main__":
    test_door_moved_rate_metric()
