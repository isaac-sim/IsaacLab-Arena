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
from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.close_door_task import CloseDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.pose import Pose, PoseRange

asset_registry = AssetRegistry()

embodiment = asset_registry.get_asset_by_name("gr1_pink")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

vegetable_type = "sweet_potato"
vegetable = asset_registry.get_asset_by_name(vegetable_type)()

# from isaaclab_arena.relations.relations import IsAnchor, On
# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.add_relation(IsAnchor())
# tomato_soup_can.add_relation(On(cracker_box))

kitchen_background = asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(style_id=2)

light = asset_registry.get_asset_by_name("light")()

# Refrigerator
refrigerator = OpenableObjectReference(
    name="refrigerator",
    prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group",
    parent_asset=kitchen_background,
    openable_joint_name="fridge_door_joint",
    openable_threshold=0.5,
)

# Refrigerator (shelf)
refrigerator_shelf = ObjectReference(
    name="refrigerator_shelf",
    prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group/Refrigerator034",
    parent_asset=kitchen_background,
)

embodiment.set_initial_pose(
    Pose(
        # position_xyz=(3.943, -1.069, 0.995),
        position_xyz=(3.943, -1.0, 0.995),
        # rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)
        # [ 0, 0, 0.5, 0.8660254 ]
        rotation_wxyz=(0.8660254, 0.0, 0.0, 0.5),
    )
)


RANDOMIZATION_HALF_RANGE_X_M = 0.04
RANDOMIZATION_HALF_RANGE_Y_M = 0.01
RANDOMIZATION_HALF_RANGE_Z_M = 0.0
vegetable.set_initial_pose(
    # Bench (no randomization)
    # Pose(position_xyz=(3.922, -0.565, 1.019), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068))
    # Bench (with randomization)
    PoseRange(
        position_xyz_min=(
            4.1 - RANDOMIZATION_HALF_RANGE_X_M,
            -0.6 - RANDOMIZATION_HALF_RANGE_Y_M,
            1.0 - RANDOMIZATION_HALF_RANGE_Z_M,
        ),
        position_xyz_max=(
            4.1 + RANDOMIZATION_HALF_RANGE_X_M,
            -0.6 + RANDOMIZATION_HALF_RANGE_Y_M,
            1.0 + RANDOMIZATION_HALF_RANGE_Z_M,
        ),
        # position_xyz_max=(
        #     3.922 + RANDOMIZATION_HALF_RANGE_X_M,
        #     -0.565 + RANDOMIZATION_HALF_RANGE_Y_M,
        #     1.019 + RANDOMIZATION_HALF_RANGE_Z_M
        # ),
        rpy_min=(0.0, 0.0, 0.0),
        rpy_max=(0.0, 0.0, 0.0),
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
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
    ):
        super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

    def get_viewer_cfg(self):
        return self.subtasks[0].get_viewer_cfg()

    def get_prompt(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None


pick_and_place_task = PickAndPlaceTask(vegetable, refrigerator_shelf, kitchen_background)
close_door_task = CloseDoorTask(refrigerator, closedness_threshold=0.05, reset_openness=0.5)

task = PutAndCloseDoorTask(subtasks=[pick_and_place_task, close_door_task])

scene = Scene(assets=[kitchen_background, refrigerator, refrigerator_shelf, vegetable, light])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=task,
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
args_cli.solve_relations = True
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()


STEPS_UNTIL_TELEPORT_VEGETABLE = 50
STEPS_UNTIL_CLOSE_DOOR = 100

# %%

# Reset frequently to check the randomizations.

# Run some zero actions.
STEPS_BETWEEN_RESETS = 2
NUM_STEPS = 200
for idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        _, _, terminated, _, _ = env.step(actions)
        if idx % STEPS_BETWEEN_RESETS == 0:
            env.reset()


# %%

# Run some zero actions.
NUM_STEPS = 200
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        if _ == STEPS_UNTIL_TELEPORT_VEGETABLE:
            # Teleport the vegetable to the shelf
            vegetable.set_object_pose(
                env,
                env_ids=None,
                pose=Pose(position_xyz=(4.625, -0.395, 1.224), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)),
            )

        if _ == STEPS_UNTIL_CLOSE_DOOR:
            # Close the refrigerator, completing the task
            refrigerator.close(env, env_ids=None, percentage=0.0)

        _, _, terminated, _, _ = env.step(actions)

        if terminated:
            print("SUCCESS!!!!!!")

# %%

# Debugging the robot joint names
num_joints = len(env.scene["robot"].joint_names)
print(f"Number of joints: {num_joints}")
print(env.scene["robot"].joint_names)

# %%

# Inspect the vegetables in the registry

from lightwheel_sdk.loader import object_loader

# Print all the vegetables in the registry
for asset in object_loader.list_registry():
    properties = asset["property"]
    if "types" in properties:
        types = properties["types"]
        if "vegetable" in types:
            print(asset)

# %%
