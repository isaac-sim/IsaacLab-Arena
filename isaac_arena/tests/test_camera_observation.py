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

import os
import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 2
HEADLESS = True


def _test_camera_observation(simulation_app) -> bool:

    from isaaclab.managers import SceneEntityCfg

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.gr1t2.gr1t2 import GR1T2Embodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras"])
    # args_cli.enable_cameras = True

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
            rotation_wxyz=(1, 0, 0, 0),
        )
    )

    scene = Scene(assets=[background, cracker_box])

    isaac_arena_environment = IsaacArenaEnvironment(
        name="packing_table_pick_and_place",
        embodiment=GR1T2Embodiment(),
        scene=scene,
        task=PickAndPlaceTask(cracker_box, background),
    )

    # Compile an IsaacLab compatible arena environment configuration
    builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.device)
                obs, _, _, _, _ = env.step(actions)
                print("obs", obs)

    return True

def test_camera_observation():
    result = run_simulation_app_function_in_separate_process(
        _test_camera_observation,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_camera_observation()
