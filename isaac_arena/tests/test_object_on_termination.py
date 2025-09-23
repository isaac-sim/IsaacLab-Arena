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

# NOTE(xinjieyao, 2025-09-23): Double the num of steps as sim.dt is changed from 0.01 to 0.005
NUM_STEPS = 40
HEADLESS = False  # True
PLOT = False


def _test_object_on_destination_termination(simulation_app) -> bool:

    from isaaclab.managers import SceneEntityCfg

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.assets.object_reference import ObjectReference
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.geometry.pose import Pose
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaac_arena.tasks.terminations import object_on_destination

    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5),
            rotation_wxyz=(1, 0, 0, 0),
        )
    )

    scene = Scene(assets=[background, cracker_box, destination_location])

    isaac_arena_environment = IsaacArenaEnvironment(
        name="kitchen",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
    )

    # Compile an IsaacLab compatible arena environment configuration
    builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    try:

        # Run some zero actions.
        forces_vec = []
        force_matrix_vec = []
        velocities_vec = []
        condition_met_vec = []
        terminated_vec = []
        sensor = env.scene.sensors["pick_up_object_contact_sensor"]
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)
                # Get the force on the pick up object.
                forces_vec.append(sensor.data.net_forces_w.clone())
                force_matrix_vec.append(sensor.data.force_matrix_w.clone())
                velocities_vec.append(env.scene[cracker_box.name].data.root_lin_vel_w.clone())

                # Try the termination.
                condition_met_vec.append(
                    object_on_destination(
                        env,
                        object_cfg=SceneEntityCfg(cracker_box.name),
                        contact_sensor_cfg=SceneEntityCfg("pick_up_object_contact_sensor"),
                    )
                )
                terminated_vec.append(terminated.item())

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    # Check that the termination condition is:
    # - not met at the start (object above the drawer)
    # - met at the end (object in the drawer)
    print("Checking the object started out of the drawer")
    assert condition_met_vec[0].item() is False, "Object started in the drawer"
    # Check if the object was in the drawer at any point.
    print("Checking the object ended in the drawer")
    assert any(condition_met_vec), "Object did not end in the drawer"
    print("Checking the task was terminated")
    assert any(terminated_vec), "The task was not terminated"
    # Check that the reset fired and moved the object above the drawer
    print("Checking the reset fired and the object was moved above the drawer")
    assert condition_met_vec[-1].item() is False, "Object was not moved above the drawer"

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
