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
import os
import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 20
HEADLESS = True
PLOT = False


def _test_object_on_destination_termination(simulation_app) -> bool:

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaac_arena.tasks.terminations import object_on_destination

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen_pick_and_place")()
    object = asset_registry.get_asset_by_name("cracker_box")()

    isaac_arena_environment = IsaacArenaEnvironment(
        name="kitchen_pick_and_place",
        embodiment=FrankaEmbodiment(),
        scene=PickAndPlaceScene(background, object),
        task=PickAndPlaceTask(),
    )

    # Set the initial pose of the object above the drawer, such that it falls in
    # and triggers the termination condition.
    position_above_drawer = Pose(
        position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
        rotation_wxyz=(1, 0, 0, 0),
    )

    isaac_arena_environment.scene.background_scene.object_pose = position_above_drawer

    # Compile an IsaacLab compatible arena environment configuration
    builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env_name, runtime_cfg = builder.build_registered()
    env = gym.make(env_name, cfg=runtime_cfg).unwrapped
    env.reset()

    # Run some zero actions.
    forces_vec = []
    force_matrix_vec = []
    velocities_vec = []
    condition_met_vec = []
    sensor = env.scene.sensors["pick_up_object_contact_sensor"]
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            # Get the force on the pick up object.
            forces_vec.append(sensor.data.net_forces_w.clone())
            force_matrix_vec.append(sensor.data.force_matrix_w.clone())
            velocities_vec.append(env.scene["pick_up_object"].data.root_lin_vel_w.clone())

            # Try the termination.
            condition_met_vec.append(object_on_destination(env))

    env.close()

    # Check that the termination condition is:
    # - not met at the start (object above the drawer)
    # - met at the end (object in the drawer)
    print("Checking the object started out of the drawer")
    assert condition_met_vec[0].item() is False, "Object started in the drawer"
    print("Checking the object ended in the drawer")
    # Check if the object was in the drawer at any point.
    assert any(condition_met_vec), "Object did not end in the drawer"

    # NOTE(alexmillane, 2025-08-04): Plot viewing is currently not working. It's complaining
    # about some non-interactive backend. For now I'm just saving the figure in the current
    # directory to a file.
    if PLOT:
        import matplotlib.pyplot as plt

        forces_np = torch.stack([torch.squeeze(f) for f in forces_vec]).cpu().numpy()
        force_matrix_np = torch.stack([torch.squeeze(f) for f in force_matrix_vec]).cpu().numpy()
        velocities_np = torch.stack([torch.squeeze(v) for v in velocities_vec]).cpu().numpy()
        condition_met_np = torch.stack([torch.squeeze(c) for c in condition_met_vec]).cpu().numpy()

        ax = plt.subplot(2, 2, 1)
        ax.plot(forces_np)
        ax.set_title("Sum of forces")
        ax = plt.subplot(2, 2, 2)
        ax.plot(force_matrix_np)
        ax.set_title("Forces against against destination drawer")
        ax = plt.subplot(2, 2, 3)
        ax.plot(velocities_np)
        ax.set_title("Object velocities")
        ax = plt.subplot(2, 2, 4)
        ax.plot(condition_met_np)
        ax.set_title("Condition met")
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "test_object_on_destination_termination.png")
        print(f"Saving plot to {plot_path}")
        plt.savefig(plot_path)

    return True


def test_object_on_destination_termination():
    result = run_simulation_app_function_in_separate_process(
        _test_object_on_destination_termination,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_on_destination_termination()
