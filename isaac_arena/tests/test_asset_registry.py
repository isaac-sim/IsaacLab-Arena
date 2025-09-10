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

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.geometry.pose import Pose
from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 2
HEADLESS = True
OBJECT_SEPARATION = 0.5


def _test_default_assets_registered(simulation_app):
    from isaac_arena.assets.asset_registry import AssetRegistry

    asset_registry = AssetRegistry()
    assert asset_registry is not None
    num_assets = len(asset_registry.components)
    print(f"Number of assets registered: {num_assets}")
    assert num_assets > 0
    num_background_assets = len(asset_registry.get_assets_by_tag("background"))
    print(f"Number of background assets registered: {num_background_assets}")
    assert num_background_assets > 0
    num_assets = len(asset_registry.get_assets_by_tag("object"))
    print(f"Number of pick up object assets registered: {num_assets}")
    assert num_assets > 0
    return True


def test_default_assets_registered():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function_in_separate_process(
        _test_default_assets_registered,
    )
    assert result, "Test failed"


def _test_all_assets_in_registry(simulation_app):
    # Import the necessary classes.
    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.assets.object import Object
    from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.dummy_task import DummyTask

    # Base Environment
    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    asset = asset_registry.get_asset_by_name("cracker_box")()

    first_position = (0.5, 0.0, 0.0)
    objects_in_registry_names: list[str] = []
    objects_in_registry: list[Object] = []
    for idx, asset_cls in enumerate(asset_registry.get_assets_by_tag("object")):
        asset = asset_cls()
        # Set their pose
        pose = Pose(
            position_xyz=(
                first_position[0] + (idx + 1) * OBJECT_SEPARATION,
                first_position[1],
                first_position[2],
            ),
            rotation_wxyz=(1, 0, 0, 0),
        )
        asset.set_initial_pose(pose)
        objects_in_registry.append(asset)
        objects_in_registry_names.append(asset.name)
    assert len(objects_in_registry) > 0

    scene = Scene(assets=[background, *objects_in_registry])

    isaac_arena_environment = IsaacArenaEnvironment(
        name="dummy_task",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=DummyTask(),
    )

    # Compile the environment.
    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    # Run
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    # Check all the assets made it into the scene.
    for asset_name in objects_in_registry_names:
        assert asset_name in env.scene.keys(), f"Asset {asset_name} not found in scene"

    # Check all the assets have the correct pose.
    for asset_name in objects_in_registry_names:
        assert asset_name in env.scene.keys(), f"Asset {asset_name} not found in scene"

    # Close the environment.
    env.close()

    return True


def test_all_assets_in_registry():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function_in_separate_process(
        _test_all_assets_in_registry,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_default_assets_registered()
    test_all_assets_in_registry()
