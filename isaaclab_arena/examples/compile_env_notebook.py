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
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.close_door_task import CloseDoorTask
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

asset_registry = AssetRegistry()

# background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("gr1_pink")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

vegetable_type = "sweet_potato"
vegetable = asset_registry.get_asset_by_name(vegetable_type)()


cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

from isaaclab_arena.assets.background_library import LibraryBackground
from lightwheel_sdk.loader import floorplan_loader

class LightwheelKitchenBackground(LibraryBackground):
    """
    Encapsulates the background scene for the kitchen.
    """

    name = "lightwheel_kitchen"
    tags = ["background"]
    # usd_path = str(floorplan_loader.acquire_usd(
    #     scene="robocasakitchen",
    #     layout_id=1,
    #     style_id=2,
    #     backend="robocasa"
    # ).result()[0])
    # initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    usd_path = None # Lazy download in the constructor
    initial_pose = Pose.identity()
    object_min_z = -0.2

    def __init__(self, layout_id: int = 1, style_id: int = 1):
        # Lazily download the USD
        self.usd_path = str(floorplan_loader.get_usd(
            scene="robocasakitchen",
            layout_id=layout_id,
            style_id=style_id,
            backend="robocasa"
        )[0])
        super().__init__()

kitchen_background = LightwheelKitchenBackground(style_id=2)

light = asset_registry.get_asset_by_name("light")()

# Refrigerator
refrigerator = OpenableObjectReference(
    name="refrigerator",
    prim_path="{ENV_REGEX_NS}/lightwheel_kitchen/fridge_main_group",
    parent_asset=kitchen_background,
    openable_joint_name="fridge_door_joint",
    openable_threshold=0.5,
)

# Refrigerator (shelf)
refrigerator_shelf = ObjectReference(
    name="refrigerator_shelf",
    prim_path="{ENV_REGEX_NS}/lightwheel_kitchen/fridge_main_group/Refrigerator034",
    parent_asset=kitchen_background,
)

embodiment.set_initial_pose(
    Pose(
        position_xyz=(3.943, -1.069, 0.995),
        rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)
    )
)

vegetable.set_initial_pose(
    # Bench
    Pose(
        position_xyz=(3.922, -0.565, 1.019),
        rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)
    )
    # Above shelf
    # Pose(
    #     position_xyz=(4.625, -0.395, 1.224),
    #     rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)
    # )
)

class PutAndCloseDoorTask(SequentialTaskBase):

    def __init__(
        self,
        # openable_object,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
    ):
        super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)
        # self.openable_object = openable_object

    def get_viewer_cfg(self):
        return self.subtasks[0].get_viewer_cfg()

    def get_prompt(self) -> str:
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

pick_and_place_task = PickAndPlaceTask(vegetable, refrigerator_shelf, kitchen_background)
close_door_task = CloseDoorTask(refrigerator, closedness_threshold=0.05, reset_openness=0.5)

task = PutAndCloseDoorTask(
    subtasks=[pick_and_place_task, close_door_task]
)


scene = Scene(assets=[kitchen_background, cracker_box, refrigerator, refrigerator_shelf, vegetable, light])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    # task=DummyTask(),
    # task=PickAndPlaceTask(vegetable, refrigerator_shelf, kitchen_background),
    task=task,
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# Open the refrigerator
# TODO: Remove. We want the robot to open this.
# refrigerator.open(env, env_ids=None, percentage=1.0)#0.4)


STEPS_UNTIL_TELEPORT_VEGETABLE = 50
STEPS_UNTIL_CLOSE_DOOR = 100

#%%

# Run some zero actions.
NUM_STEPS = 200
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        if _ == STEPS_UNTIL_TELEPORT_VEGETABLE:
            # Teleport the vegetable to the shelf
            vegetable.set_object_pose(env, env_ids=None, pose=Pose(position_xyz=(4.625, -0.395, 1.224), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)))

        if _ == STEPS_UNTIL_CLOSE_DOOR:
            # Close the refrigerator, completing the task
            refrigerator.close(env, env_ids=None, percentage=0.0)

        _, _, terminated, _, _ = env.step(actions)
        if terminated:
            print("SUCCESS!!!!!!")

        # if terminated:
        #     refrigerator.open(env, env_ids=None, percentage=1.0)#0.4)


# %%

# Inspect the vegetables in the registry

from lightwheel_sdk.loader import object_loader

# Print all the vegetables in the registry
for asset in object_loader.list_registry():
    properties = asset['property']
    if 'types' in properties:
        types = properties['types']
        if 'vegetable' in types:
            print(asset)

#%%


