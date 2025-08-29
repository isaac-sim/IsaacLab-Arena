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
from collections.abc import Callable

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 10
HEADLESS = True


def _test_open_door_microwave(simulation_app) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.open_door_scene import OpenDoorScene
    from isaac_arena.tasks.open_door_task import OpenDoorTask

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
    microwave = asset_registry.get_asset_by_name("microwave")()

    # Put the microwave on the packing table.
    microwave.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    isaac_arena_environment = IsaacArenaEnvironment(
        name="open_door",
        embodiment=FrankaEmbodiment(),
        scene=OpenDoorScene(background, microwave),
        task=OpenDoorTask(),
    )

    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    def step_zeros_and_call(env: ManagerBasedEnv, function: Callable[[ManagerBasedEnv], None], num_steps: int):
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.device)
                env.step(actions)
                function(env)

    def assert_closed(env: ManagerBasedEnv):
        is_open = microwave.is_open(env, "interactable_object")
        assert is_open.shape == torch.Size([1])
        assert not is_open.item()
        if not is_open.item():
            print("Microwave is closed")

    def assert_open(env: ManagerBasedEnv):
        is_open = microwave.is_open(env, "interactable_object")
        assert is_open.shape == torch.Size([1])
        assert is_open.item()
        if is_open.item():
            print("Microwave is open")

    print("Closing microwave")
    microwave.close(env, "interactable_object")
    step_zeros_and_call(env, assert_closed, NUM_STEPS)
    print("Opening microwave")
    microwave.open(env, "interactable_object")
    step_zeros_and_call(env, assert_open, NUM_STEPS)

    env.close()

    return True


def test_open_door_microwave():
    result = run_simulation_app_function_in_separate_process(
        _test_open_door_microwave,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_open_door_microwave()
