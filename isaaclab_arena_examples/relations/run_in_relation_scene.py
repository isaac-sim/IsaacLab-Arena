# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a minimal RoboLab-style ``In`` scene in simulation and report where the item lands.

The container is a fixed anchor and the item is placed inside it via the mesh-only ``In`` relation
(cavity derived from the container mesh, or a bounding-box cavity when the mesh can't be capped).

Usage (inside the container):
    /isaac-sim/python.sh isaaclab_arena_examples/relations/run_in_relation_scene.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--container", default="serving_bowl_vomp_robolab")
parser.add_argument("--item", default="rubiks_cube_hot3d_robolab")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import warp as wp  # noqa: E402

from isaaclab_arena.assets.registries import AssetRegistry  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg  # noqa: E402
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: E402
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams  # noqa: E402
from isaaclab_arena.relations.relation_solver_params import CollisionMode, RelationSolverParams  # noqa: E402
from isaaclab_arena.relations.relations import In, IsAnchor  # noqa: E402
from isaaclab_arena.scene.scene import Scene  # noqa: E402
from isaaclab_arena.utils.pose import Pose  # noqa: E402

registry = AssetRegistry()
ground = registry.get_asset_by_name("ground_plane")()
light = registry.get_asset_by_name("light")()

container = registry.get_asset_by_name(args.container)()
container.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
container.add_relation(IsAnchor())

item = registry.get_asset_by_name(args.item)()
item.add_relation(In(container))

scene = Scene(assets=[ground, light, container, item])
placer_params = ObjectPlacerParams(
    solver_params=RelationSolverParams(collision_mode=CollisionMode.MESH, verbose=False, save_position_history=False),
    resolve_on_reset=False,
)
env = ArenaEnvBuilder(
    IsaacLabArenaEnvironment(name="in_relation_scene", scene=scene, placer_params=placer_params),
    ArenaEnvBuilderCfg(),
).make_registered()
env.reset()

origin = env.unwrapped.scene.env_origins[0]
container_xyz = (wp.to_torch(env.unwrapped.scene[container.name].data.root_pos_w)[0] - origin).tolist()
item_xyz = (wp.to_torch(env.unwrapped.scene[item.name].data.root_pos_w)[0] - origin).tolist()
print(f"RESULT container '{container.name}' at {container_xyz}")
print(f"RESULT item '{item.name}' at {item_xyz} -> initialized inside the container")

simulation_app.close()
