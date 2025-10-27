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

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 2
HEADLESS = True
ENABLE_CAMERAS = True


def _test_camera_observation(simulation_app) -> bool:

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2PinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
            rotation_wxyz=(1, 0, 0, 0),
        )
    )

    scene = Scene(assets=[background, cracker_box])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="camera_observation_test",
        embodiment=GR1T2PinkEmbodiment(enable_cameras=True),
        scene=scene,
        task=DummyTask(),
    )

    # Compile an IsaacLab compatible arena environment configuration
    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None
    env.reset()

    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            obs, _, _, _, _ = env.step(actions)
            # Get the camera observation
            camera_observation = obs["camera_obs"]["robot_pov_cam_rgb"]
            # Assert that the camera rgb observation has three channels
            assert camera_observation.shape[3] == 3, "Camera rgb observation does not have three channels"
            # Make sure the camera observation contains values other than 0
            assert camera_observation.any() != 0, "Camera observation contains only 0s"

    env.close()

    return True


@pytest.mark.with_cameras
def test_camera_observation():
    # Safety: if someone runs this without the env var, skip with a helpful message
    if not ENABLE_CAMERAS:
        pytest.skip("This test requires cameras. Re-run with: WITH_CAMERAS=1 pytest -m with_cameras")

    result = run_simulation_app_function(
        _test_camera_observation,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_camera_observation()
