# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example notebook demonstrating ObjectPlacer with real Isaac Sim objects."""

# %%
# NOTE: When running as a notebook, first run this cell to launch the simulation app:
# import pinocchio  # noqa: F401
# from isaaclab.app import AppLauncher
# print("Launching simulation app once in notebook")
# simulation_app = AppLauncher()

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def run_isaac_sim_object_placer_demo(num_steps: int = 10000):
    """Run the ObjectPlacer demo with Isaac Sim objects.

    Args:
        num_steps: Number of simulation steps to run.
    """
    import torch
    import tqdm

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    # Create objects from asset registry
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    light = asset_registry.get_asset_by_name("light")()

    # Table as anchor
    office_table = asset_registry.get_asset_by_name("office_table")()
    office_table.set_initial_pose(Pose(position_xyz=(1.0, 1.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    # Central object on the table
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    # Objects placed on each side of cracker_box
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()  # RIGHT side
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()  # LEFT side
    mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()  # FRONT side
    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()  # BACK side

    # Object on top of cracker_box
    mug = asset_registry.get_asset_by_name("mug")()

    # Mark office_table as the anchor for relation solving
    office_table.add_relation(IsAnchor())

    # Central object on the table
    cracker_box.add_relation(On(office_table, clearance_m=0.02))

    # Objects placed on each side of cracker_box (all on table surface)
    dex_cube.add_relation(On(office_table, clearance_m=0.02))
    dex_cube.add_relation(NextTo(cracker_box, side=Side.RIGHT, distance_m=0.05))

    sugar_box.add_relation(On(office_table, clearance_m=0.02))
    sugar_box.add_relation(NextTo(cracker_box, side=Side.LEFT, distance_m=0.05))

    mustard_bottle.add_relation(On(office_table, clearance_m=0.02))
    mustard_bottle.add_relation(NextTo(cracker_box, side=Side.FRONT, distance_m=0.05))

    tomato_soup_can.add_relation(On(office_table, clearance_m=0.02))
    tomato_soup_can.add_relation(NextTo(cracker_box, side=Side.BACK, distance_m=0.05))

    # Mug on top of cracker_box
    mug.add_relation(On(cracker_box, clearance_m=0.02))

    # Build and run the environment
    # Note: ArenaEnvBuilder automatically solves relations via ObjectPlacer
    assets = [
        ground_plane,
        light,
        office_table,
        cracker_box,
        dex_cube,
        sugar_box,
        mustard_bottle,
        tomato_soup_can,
        mug,
    ]
    scene = Scene(assets=assets)
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="relation_all_sides_demo",
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    # Run simulation steps
    for _ in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)


def smoke_test_isaac_sim_object_placer(simulation_app: SimulationApp) -> bool:
    """Smoke test: run ObjectPlacer with Isaac Sim objects (minimal steps)."""
    run_isaac_sim_object_placer_demo(num_steps=2)
    return True


# %%
# When running as a notebook (after launching simulation_app), uncomment and run:
# run_isaac_sim_object_placer_demo()

# %%
