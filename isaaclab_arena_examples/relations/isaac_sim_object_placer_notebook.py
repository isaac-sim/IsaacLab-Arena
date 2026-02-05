# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Example notebook demonstrating ObjectPlacer with real Isaac Sim objects."""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacsim import SimulationApp


def run_isaac_sim_object_placer_demo(
    num_steps: int = 10000,
    reset_every_n_steps: int = 100,
):
    """Run the ObjectPlacer demo with Isaac Sim objects.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset the environment every N steps (to see RandomAroundSolution in action).
    """
    import torch
    import tqdm

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor, NextTo, On, RandomAroundSolution, Side
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    table_background = asset_registry.get_asset_by_name("office_table")()
    light = asset_registry.get_asset_by_name("light")()

    tabletop_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
        parent_asset=table_background,
    )
    # Mark the ObjectReference as the anchor for relation solving (not subject to optimization).
    tabletop_reference.add_relation(IsAnchor())

    # Put a cracker box on the counter.
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    cracker_box.add_relation(On(tabletop_reference, clearance_m=0.02))
    # Place the cracker box explicitly slightly to the right of the tabletop.
    cracker_box.add_relation(AtPosition(x=-0.1, y=0.0))
    cracker_box.add_relation(RandomAroundSolution(x_half_m=0.05, y_half_m=0.25))

    # Put a mug next to the cracker box.
    mug = asset_registry.get_asset_by_name("mug")()
    mug.add_relation(On(tabletop_reference, clearance_m=0.02))
    mug.add_relation(NextTo(cracker_box, side=Side.RIGHT, distance_m=0.1))

    tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
    tomato_soup_can.add_relation(On(tabletop_reference, clearance_m=0.02))
    tomato_soup_can.add_relation(NextTo(cracker_box, side=Side.LEFT, distance_m=0.1))

    scene = Scene(assets=[ground_plane, table_background, tabletop_reference, cracker_box, mug, tomato_soup_can, light])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="isaac_sim_object_placer_demo",
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    # Run simulation steps with periodic resets
    for step in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        # Reset every N steps to see RandomAroundSolution randomization
        if reset_every_n_steps > 0 and (step + 1) % reset_every_n_steps == 0:
            env.reset()


def smoke_test_isaac_sim_object_placer(simulation_app: SimulationApp) -> bool:
    """Smoke test: run ObjectPlacer with Isaac Sim objects (minimal steps)."""
    run_isaac_sim_object_placer_demo(num_steps=2)
    return True


# %%
# When running as a notebook (after launching simulation_app), uncomment and run:
run_isaac_sim_object_placer_demo()

# %%
