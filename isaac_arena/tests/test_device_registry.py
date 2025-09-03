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

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 2
HEADLESS = True
DEVICE_NAMES = ["avp_handtracking", "spacemouse", "keyboard"]


def _test_all_devices_in_registry(simulation_app):
    # Import the necessary classes.
    # Needed for sim app restart.
    import omni.usd

    from isaac_arena.assets.asset_registry import AssetRegistry, DeviceRegistry
    from isaac_arena.embodiments.gr1t2.gr1t2 import GR1T2Embodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

    # Base Environment
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
    asset = asset_registry.get_asset_by_name("cracker_box")()

    for device_name in DEVICE_NAMES:
        # We do a reset to start sim only once.
        omni.usd.get_context().new_stage()

        teleop_device = device_registry.get_device_by_name(device_name)()
        isaac_arena_environment = IsaacArenaEnvironment(
            name="kitchen_pick_and_place",
            embodiment=GR1T2Embodiment(),
            scene=Scene([background, asset]),
            task=PickAndPlaceTask(pick_up_object=asset, background_scene=background),
            teleop_device=teleop_device,
        )

        # Remove previous environment if it exists.
        if isaac_arena_environment.name in gym.registry:
            del gym.registry[isaac_arena_environment.name]

        # Compile the environment.
        args_parser = get_isaac_arena_cli_parser()
        args_cli = args_parser.parse_args(["--teleop_device", teleop_device.name])

        builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)

        env = builder.make_registered()

        # disable control on stop
        env.sim._app_control_on_stop_handle = None  # type: ignore

        env.reset()
        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

    # Close the environment.
    env.close()

    return True


def test_all_devices_in_registry():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function_in_separate_process(
        _test_all_devices_in_registry,
        headless=HEADLESS,
    )
    assert result, "Test failed"
