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

import gymnasium as gym
import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.assets.object_reference import ObjectReference
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.g1.g1 import G1Embodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("galileo")()
    pick_up_object = asset_registry.get_asset_by_name("power_drill")()
    pick_up_object.set_initial_pose(
        Pose(
            position_xyz=(0.55, 0.0, 0.33),
            rotation_wxyz=(0.0, 0.0, 0.7071068, -0.7071068),
        )
    )

    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/galileo/BackgroundAssets/bins/small_bin_grid_01/lid",
        parent_asset=background,
    )
    scene = Scene(assets=[background])

    isaac_arena_environment = IsaacArenaEnvironment(
        name="pick_and_place",
        embodiment=G1Embodiment(),
        scene=scene,
        task=PickAndPlaceTask(pick_up_object, destination_location, background),
    )

    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env


def step_standing_actions_call(env, num_steps, robot_init_base_pose, function=None):
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            actions[:, -4] = 0.75
            _, _, _, _, _ = env.step(actions)
            if function is not None:
                function(env, robot_init_base_pose)


def _test_standing_idle_actions(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    # Get the scene
    env = get_test_environment(num_envs=1)

    def assert_standing_idle(env: ManagerBasedEnv, robot_init_base_pose: torch.Tensor):

        # get robot base pose after actions call
        robot_base_pose = env.scene.robot.get_base_pose()
        print(f"robot_base_pose: {robot_base_pose}")
        print(f"robot_init_base_pose: {robot_init_base_pose}")
        # check if robot base pose is close to initial base pose
        assert torch.allclose(robot_base_pose, robot_init_base_pose, atol=1e-2)

    try:
        # get robot init base pose
        robot_init_base_pose = env.scene.robot.get_init_base_pose()
        # sending standing idle actions
        step_standing_actions_call(env, NUM_STEPS, robot_init_base_pose, assert_standing_idle)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_standing_idle_actions_single_env():
    result = run_simulation_app_function_in_separate_process(
        _test_standing_idle_actions,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_standing_idle_actions.__name__} failed"


if __name__ == "__main__":
    test_standing_idle_actions_single_env()
