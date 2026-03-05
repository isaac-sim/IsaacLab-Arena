# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import argparse
import torch
import tqdm

from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--visualizer", "kit"])
# args = parser.parse_args([])
app_launcher = AppLauncher(args)

# %%

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
# embodiment = asset_registry.get_asset_by_name("franka")()
embodiment = asset_registry.get_asset_by_name("franka")()
# embodiment = asset_registry.get_asset_by_name("gr1_pink")(enable_cameras=True)
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
# tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
microwave = asset_registry.get_asset_by_name("microwave")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

table_top_reference = ObjectReference(
    name="table_top_reference",
    prim_path="{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static/Counter_Top_A",
    parent_asset=background,
)
table_top_reference.add_relation(IsAnchor())

microwave.add_relation(AtPosition(x=0.4, y=0.0))
microwave.add_relation(On(table_top_reference))

cracker_box.add_relation(On(microwave))


destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)

scene = Scene(assets=[background, table_top_reference, microwave, destination_location, cracker_box])
# scene = Scene(assets=[background, cracker_box, tomato_soup_can])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, destination_location, background),
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
args_cli.solve_relations = True
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 300
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%

from isaaclab_arena.utils.reload_modules import reload_arena_modules

reload_arena_modules()


# %%

from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app

teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
