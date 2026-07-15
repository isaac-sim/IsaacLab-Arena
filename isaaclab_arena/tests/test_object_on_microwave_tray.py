# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the microwave-tray contact fires a pick-and-place success termination."""

import torch
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 120
HEADLESS = True


def _test_object_on_microwave_tray_termination(simulation_app) -> bool:
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()

    microwave.set_initial_pose(
        Pose(position_xyz=(0.4, -0.00586, 0.22773), rotation_xyzw=(0.0, 0.0, -0.7071068, 0.7071068))
    )

    # Destination reference targeting the microwave turntable rigid body (the filter under test).
    destination_ref = ObjectReference(
        name="microwave_disc",
        parent_asset=microwave,
        prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
        object_type=ObjectType.RIGID,
    )

    scene = Scene(assets=[background, microwave, dex_cube])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="microwave_tray",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
        task=PickAndPlaceTask(dex_cube, destination_ref, background),
    )

    env = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    try:
        # Teleport the cube just above the tray and drop it (zero velocity, zero actions).
        # Tray world position (microwave x/y) plus a 0.06 m drop height.
        cube_asset = env.unwrapped.scene[dex_cube.name]
        target_pos = torch.tensor([0.4, -0.00586, 0.28773], device=env.unwrapped.device)
        root_pose = torch.zeros((1, 7), device=env.unwrapped.device)
        root_pose[0, :3] = target_pos
        root_pose[0, 3] = 1.0  # identity quaternion (w, x, y, z)
        cube_asset.write_root_pose_to_sim(root_pose)
        cube_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))

        # Open the microwave door so the cube drops onto the tray.
        microwave.open(env, env_ids=None)

        success_vec = []
        terminated_vec = []
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(NUM_STEPS):
            with torch.inference_mode():
                _, _, terminated, _, _ = env.step(actions)
                success_vec.append(env.unwrapped.termination_manager.get_term("success").clone())
                terminated_vec.append(terminated.item())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("Checking the cube was not on the tray at the first step")
    assert not success_vec[0].item(), "Cube registered success before it could fall onto the tray"
    print("Checking the cube landed on the tray and fired the success termination")
    assert any(s.item() for s in success_vec), "Cube on the tray never fired the success termination"
    print("Checking the task terminated")
    assert any(terminated_vec), "The task was not terminated"

    return True


def test_object_on_microwave_tray_termination():
    result = run_simulation_app_function(_test_object_on_microwave_tray_termination, headless=HEADLESS)
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_on_microwave_tray_termination()
