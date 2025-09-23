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

import numpy as np
import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 10
HEADLESS = True
STANDING_POSITION_XY_EPS = 1e-1


def get_test_environment(num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.g1.g1 import G1Embodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.dummy_task import DummyTask

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()

    scene = Scene(assets=[background])
    embodiment = G1Embodiment()
    # NOTE(xinjieyao, 2025.09.22): Set initial pose such that robot will not drop to the ground, causing WBC unstable.
    robot_init_base_pose = np.array([0, 0, 0])
    embodiment.set_initial_pose(Pose(position_xyz=robot_init_base_pose, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    isaac_arena_environment = IsaacArenaEnvironment(
        name="g1_standing_test",
        embodiment=embodiment,
        scene=scene,
        task=DummyTask(),
    )

    args_cli = get_isaac_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    return env, robot_init_base_pose


def step_standing_actions_call(env, num_steps, robot_init_base_pose, function=None):
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            # NOTE(xinjieyao, 2025.09.22): Set base height to 0.75m to avoid robot squatting to match 0-height command.
            actions[:, -4] = 0.75
            _, _, _, _, _ = env.step(actions)
            if function is not None:
                function(env, robot_init_base_pose)


def _test_standing_idle_actions(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    # Get the scene
    env, robot_init_base_pose = get_test_environment(num_envs=1)

    def assert_standing_idle(env: ManagerBasedEnv, robot_init_base_pose: np.ndarray):
        # get robot base pose after actions call
        robot_base_pose = env.scene["robot"].data.root_link_pose_w[0, :3].cpu().numpy()
        # check if robot base pose is close to initial base pose
        robot_xy_error = np.linalg.norm(robot_base_pose[:2] - robot_init_base_pose[:2])
        assert robot_xy_error < STANDING_POSITION_XY_EPS, "Robot moved away from initial position."

    try:
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
