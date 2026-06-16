# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

args_cli = get_isaaclab_arena_cli_parser().parse_args(["--viz", "kit", "--enable_cameras"])
print("Launching simulation app once in notebook")
simulation_app = AppLauncher(args_cli)


# %%

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
franka = asset_registry.get_asset_by_name("franka_ik")(enable_cameras=True)
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
dome_light = asset_registry.get_asset_by_name("light")()

cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
cracker_box.add_relation(IsAnchor())
tomato_soup_can.add_relation(On(cracker_box))

scene = Scene(assets=[background, cracker_box, tomato_soup_can, dome_light])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=franka,
    scene=scene,
)

dome_light.get_variation("hdr_image").enable()
franka.get_variation("camera_extrinsics_wrist_cam").enable()

env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
print(env_builder.get_variations_catalogue_as_string())

env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
RESET_ON_EVERY_N_STEPS = 10
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
    if _ % RESET_ON_EVERY_N_STEPS == 0:
        env.reset()

# %%

from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules

# Run this to tear down the simulation app, for rebuilding the environment, without requiring a restart.
reload_arena_modules()
teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

# %%
