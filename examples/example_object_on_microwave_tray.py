# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize a cube falling onto the microwave tray in Omniverse Kit."""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and enable the Kit visualizer by default."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=120,
        help="Maximum number of simulation steps to wait for the cube to reach the tray.",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the Kit window open after the demonstration finishes.",
    )
    parser.set_defaults(num_envs=1, visualizer=["kit"])
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """Run the microwave-tray contact demonstration."""
    import torch

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

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
        cube_asset.write_root_pose_to_sim_index(root_pose=root_pose)
        cube_asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=env.unwrapped.device))

        # Open the microwave door so the cube drops onto the tray.
        microwave.open(env, env_ids=None)

        succeeded = False
        terminated = False
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                _, _, terminated_tensor, _, _ = env.step(actions)
                succeeded = succeeded or env.unwrapped.termination_manager.get_term("success").item()
                terminated = terminated or terminated_tensor.item()
            time.sleep(env.unwrapped.step_dt)

        assert succeeded, "Cube on the tray never fired the success termination"
        assert terminated, "The task was not terminated"
        print("Cube landed on the microwave tray and fired the success termination.")

        if args_cli.keep_open:
            print("Keeping the Kit window open. Close it or press Ctrl+C to exit.")
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
