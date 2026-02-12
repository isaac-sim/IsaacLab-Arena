# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

# object_set = RigidObjectSet(
#     name="object_set",
#     objects=[cracker_box, cracker_box],
# )
OBJECT_SET_1_PRIM_PATH = "/World/envs/env_.*/ObjectSet_1"
# object_set = RigidObjectSet(
#     name="single_object_set", objects=[cracker_box, cracker_box], prim_path=OBJECT_SET_1_PRIM_PATH
# )

# # scene = Scene(assets=[background, cracker_box, tomato_soup_can])
# scene = Scene(assets=[background, object_set])
# isaaclab_arena_environment = IsaacLabArenaEnvironment(
#     name="reference_object_test",
#     embodiment=embodiment,
#     scene=scene,
# )

asset_registry = AssetRegistry()
background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)
obj_set = RigidObjectSet(
    name="single_object_set", objects=[cracker_box, tomato_soup_can], prim_path=OBJECT_SET_1_PRIM_PATH
)
obj_set.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
scene = Scene(assets=[background, obj_set])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="single_object_set_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(
        pick_up_object=obj_set, destination_location=destination_location, background_scene=background
    ),
    teleop_device=None,
)


args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "3"])
args_cli.solve_relations = True
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 500
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
