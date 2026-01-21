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
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

initial_pose = Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
cracker_box.set_initial_pose(initial_pose)


from isaaclab_arena.utils.pose import PoseRange

pose_range = PoseRange(
    position_xyz_min=(
        initial_pose.position_xyz[0] - 0.08,
        initial_pose.position_xyz[1] - 0.08,
        initial_pose.position_xyz[2],
    ),
    position_xyz_max=(
        initial_pose.position_xyz[0] + 0.08,
        initial_pose.position_xyz[1] + 0.08,
        initial_pose.position_xyz[2],
    ),
    rpy_min=(0.0, 0.0, 0.0),
    rpy_max=(0.0, 0.0, 0.0),
)
print(f"pose_range: {pose_range}")
print(f"pose_range.get_midpoint(): {pose_range.get_midpoint()}")

cracker_box.set_initial_pose(pose_range)
# cracker_box.set_initial_pose(initial_pose)

from isaaclab_arena.tasks.lift_object_task import LiftObjectTask
task = LiftObjectTask(cracker_box, background, reset_pose_range=pose_range)
# task = DummyTask()

scene = Scene(assets=[background, cracker_box])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    # task=DummyTask(),
    task=task,
    teleop_device=None,
    # env_cfg_callback=test_callback,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

with torch.inference_mode():
    env.reset()

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
