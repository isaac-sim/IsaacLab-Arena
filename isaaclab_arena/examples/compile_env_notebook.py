# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

print(f"ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

from isaaclab_arena.assets.background import Background


class ObjectReferenceTestKitchenBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            name="kitchen",
            tags=["background", "pick_and_place"],
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/reference_object_test_kitchen.usd",
            # initial_pose=Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711)),
            initial_pose=Pose(position_xyz=(1.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
            object_min_z=-0.2,
        )


background = ObjectReferenceTestKitchenBackground()
# background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
embodiment = asset_registry.get_asset_by_name("franka")()
# cracker_box = asset_registry.get_asset_by_name("cracker_box")()
destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)

from isaaclab_arena.assets.object_base import ObjectType

cracker_box = ObjectReference(
    name="cracker_box",
    prim_path="{ENV_REGEX_NS}/kitchen/_03_cracker_box",
    parent_asset=background,
    object_type=ObjectType.RIGID,
)

scene = Scene(assets=[background, cracker_box, destination_location])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %

# Run some zero actions.
NUM_STEPS = 50
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        _, _, terminated, _, _ = env.step(actions)
        print(f"terminated: {terminated}")
        print(f"success: {env.termination_manager.get_term('success')}")

# %%

# %%
