# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

# Example notebook demonstrating ObjectPlacer with real Isaac Sim objects.

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.embodiments.null_embodiment import NullEmbodiment
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.relations import NextTo, On, Side
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

# %%

# Create objects from asset registry
ground_plane = asset_registry.get_asset_by_name("ground_plane")()
light = asset_registry.get_asset_by_name("light")()

office_table = asset_registry.get_asset_by_name("office_table")()
office_table.set_initial_pose(Pose(position_xyz=(1.0, 1.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

mug = asset_registry.get_asset_by_name("mug")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
coffee_machine = asset_registry.get_asset_by_name("coffee_machine")()

# Define spatial relations
coffee_machine.add_relation(On(office_table, clearance_m=0.02))
cracker_box.add_relation(On(office_table, clearance_m=0.02))
cracker_box.add_relation(NextTo(coffee_machine, side=Side.RIGHT, distance_m=0.15))
mug.add_relation(On(coffee_machine, clearance_m=0.02))

# Place objects using ObjectPlacer (handles random init, solving, and applying positions)
placer = ObjectPlacer()
result = placer.place(
    objects=[office_table, coffee_machine, cracker_box, mug],
    anchor_object=office_table,
)

# %%

assets = [ground_plane, office_table, cracker_box, coffee_machine, light, mug]
scene = Scene(assets=assets)
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=NullEmbodiment(),
    scene=scene,
    task=DummyTask(),
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# Run some zero actions.
NUM_STEPS = 10000
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
