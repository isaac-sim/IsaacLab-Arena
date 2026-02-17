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
from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

object_set = RigidObjectSet(
    name="object_set",
    objects=[cracker_box, tomato_soup_can],
)
# Different pose per object (object name -> Pose)
object_set.set_initial_pose({
    cracker_box.name: Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    tomato_soup_can.name: Pose(position_xyz=(0.35, 0.50, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
})

# scene = Scene(assets=[background, cracker_box, tomato_soup_can])
scene = Scene(assets=[background, object_set])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
# args_cli.solve_relations = True
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
name, cfg = env_builder.build_registered()

env = env_builder.make_registered(cfg)
env.reset()

# %%
# Run some zero actions.
RESET_EVERY_N_STEPS = 100
NUM_STEPS = 500
for idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        if idx % RESET_EVERY_N_STEPS == 0:
            env.reset()


# %%


# %%
