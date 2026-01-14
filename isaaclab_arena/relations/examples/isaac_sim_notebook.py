# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# Objects instead of DummyObjects
# Bounding boxes work, visualize?

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
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()


from isaaclab_arena.relations.relation_solver import RelationSolver

# %%
from isaaclab_arena.relations.relations import NextTo, On, Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, get_random_pose_within_bounding_box

ground_plane = asset_registry.get_asset_by_name("ground_plane")()
light = asset_registry.get_asset_by_name("light")()

office_table = asset_registry.get_asset_by_name("office_table")()
# The office table will be the anchor object for the relation solver and we need to set its initial pose to a fixed position.
office_table.set_initial_pose(Pose(position_xyz=(1.0, 1.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

mug = asset_registry.get_asset_by_name("mug")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
coffee_machine = asset_registry.get_asset_by_name("coffee_machine")()

# Note: We need to set the initial poses of all objects at random before running the relation solver.
# For now we set this here, but will be encapsulated in the relation solver
workspace = AxisAlignedBoundingBox(min_point=(-3.0, -3.0, 0.0), max_point=(3.0, 3.0, 1.0))
random_pose = get_random_pose_within_bounding_box(workspace)
cracker_box.set_initial_pose(random_pose)
random_pose = get_random_pose_within_bounding_box(workspace)
coffee_machine.set_initial_pose(random_pose)
random_pose = get_random_pose_within_bounding_box(workspace)
mug.set_initial_pose(random_pose)


# Set the actual relation.
coffee_machine.add_relation(On(office_table, clearance_m=0.02))
cracker_box.add_relation(On(office_table, clearance_m=0.02))
cracker_box.add_relation(NextTo(coffee_machine, side=Side.RIGHT, distance_m=0.15))
mug.add_relation(On(coffee_machine, clearance_m=0.02))


assets = [ground_plane, office_table, cracker_box, coffee_machine, light, mug]

relation_solver = RelationSolver(anchor_object=office_table)
object_positions = relation_solver.solve(objects=[office_table, coffee_machine, cracker_box, mug])

# Update the positions of the objects
for obj, pos in object_positions.items():
    if obj.name == "office_table":
        continue
    print(f"Setting optimized pose for {obj.name} to {pos}")
    obj.set_initial_pose(Pose(position_xyz=pos, rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))


# %%

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

# %%
# Visualize bounding boxes for objects
# from isaaclab_arena.utils.isaac_sim_debug_draw import IsaacSimDebugDraw
#
# debug_draw = IsaacSimDebugDraw()
# debug_draw.clear()
# debug_draw.draw_object_bboxes(objects=[office_table, cracker_box, mug])

# %%
# TODO(cvolk): Why do I need to run this cell here?
# Run some zero actions.
NUM_STEPS = 10000
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
