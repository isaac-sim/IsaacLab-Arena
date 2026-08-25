# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the microwave-tray contact fires a pick-and-place success termination."""

import torch
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 120
NUM_ENVS = 2
HEADLESS = True


def _test_object_on_microwave_tray_termination(simulation_app) -> bool:
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(NUM_ENVS)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()

    microwave.set_initial_pose(
        Pose(position_xyz=(0.4, -0.00586, 0.22773), rotation_xyzw=(0.0, 0.0, -0.7071068, 0.7071068))
    )

    # The articulation owns this rigid body's live state; the scene reference is read-only.
    destination_ref = ObjectReference(
        name="microwave_disc",
        parent_asset=microwave,
        prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
    )

    scene = Scene(assets=[background, microwave, dex_cube, destination_ref])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="microwave_tray",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(dex_cube, destination_ref, background),
    )

    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    try:
        assert destination_ref.name in env.unwrapped.scene.extras
        assert destination_ref.name not in env.unwrapped.scene.rigid_objects

        # Teleport the cube just above the tray and drop it (zero velocity, zero actions).
        # Tray world position (microwave x/y) plus a 0.06 m drop height.
        cube_asset = env.unwrapped.scene[dex_cube.name]
        target_position = torch.tensor([0.4, -0.00586, 0.28773], device=env.unwrapped.device)
        root_pose = torch.zeros((NUM_ENVS, 7), device=env.unwrapped.device)
        root_pose[:, :3] = target_position + env.unwrapped.scene.env_origins
        root_pose[:, 6] = 1.0  # identity quaternion (x, y, z, w)
        cube_asset.write_root_pose_to_sim(root_pose)
        cube_asset.write_root_velocity_to_sim(torch.zeros((NUM_ENVS, 6), device=env.unwrapped.device))

        # Open the microwave door so the cube drops onto the tray.
        microwave.open(env, env_ids=None)

        success_vec = []
        terminated_vec = []
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(NUM_STEPS):
            with torch.inference_mode():
                _, _, terminated, _, _ = env.step(actions)
                success_vec.append(env.unwrapped.termination_manager.get_term("success").clone())
                terminated_vec.append(terminated.clone())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("Checking the cube was not on the tray at the first step")
    assert not success_vec[0].any(), "Cube registered success before it could fall onto the tray"
    print("Checking the cube landed on the tray and fired the success termination")
    assert torch.stack(success_vec).any(dim=0).all(), "Cube on the tray never fired success in every environment"
    print("Checking the task terminated")
    assert torch.stack(terminated_vec).any(dim=0).all(), "The task did not terminate in every environment"

    return True


def test_object_on_microwave_tray_termination():
    result = run_function_with_persistent_simulation_app(_test_object_on_microwave_tray_termination, headless=HEADLESS)
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_on_microwave_tray_termination()
