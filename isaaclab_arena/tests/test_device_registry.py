# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function, safe_teardown

NUM_STEPS = 2
HEADLESS = True
DEVICE_NAMES = ["avp_handtracking", "spacemouse", "keyboard"]


def _test_all_devices_in_registry(simulation_app):
    # Import the necessary classes.

    from isaaclab_arena.assets.asset_registry import AssetRegistry, DeviceRegistry
    from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2PinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask

    # Base Environment
    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    asset = asset_registry.get_asset_by_name("cracker_box")()

    for device_name in DEVICE_NAMES:

        teleop_device = device_registry.get_device_by_name(device_name)()
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="kitchen",
            embodiment=GR1T2PinkEmbodiment(),
            scene=Scene([background, asset]),
            task=DummyTask(),
            teleop_device=teleop_device,
        )

        # Remove previous environment if it exists.
        if isaaclab_arena_environment.name in gym.registry:
            del gym.registry[isaaclab_arena_environment.name]

        # Compile the environment.
        args_parser = get_isaaclab_arena_cli_parser()
        args_cli = args_parser.parse_args([])

        builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)

        env = builder.make_registered()

        env.reset()
        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Close the environment using safe teardown
        # Also creates a new stage for the next test
        safe_teardown()

    return True


def test_all_devices_in_registry():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function(
        _test_all_devices_in_registry,
        headless=HEADLESS,
    )
    assert result, "Test failed"
