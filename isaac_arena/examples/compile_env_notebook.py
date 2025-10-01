# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.open_door_task import OpenDoorTask
from isaac_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
# cracker_box = asset_registry.get_asset_by_name("cracker_box")()
microwave = asset_registry.get_asset_by_name("microwave")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
microwave.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

task = OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2)

scene = Scene(assets=[background, microwave])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=task,
    teleop_device=None,
)


#%

# from dataclasses import MISSING

# from isaaclab.managers.recorder_manager import RecorderTerm
# from isaaclab.managers.recorder_manager import RecorderTermCfg
# from isaaclab.envs import ManagerBasedEnv
# from isaaclab.utils import configclass
# from isaac_arena.metrics.metric_base import MetricBase
# from isaac_arena.metrics.object_moved import ObjectVelocityRecorder
# from isaac_arena.affordances.openable import Openable
# from isaac_arena.assets.object_base import ObjectBase

# import numpy as np


# from isaac_arena.metrics.success_rate import SuccessRateMetric

# def get_metrics():
#     return [SuccessRateMetric(), DoorMovedRateMetric(microwave, reset_openness=task.reset_openness)]

# # Replace the existing metrics the new ones.
# isaac_arena_environment.task.get_metrics = get_metrics
# print(isaac_arena_environment.task.get_metrics())

#%

args_cli = get_isaac_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env_cfg = env_builder.compose_manager_cfg()
env_cfg.episode_length_s = 0.10
env_cfg.terminations.time_out.time_out = True
# env = env_builder.make_registered()
env = env_builder.make_registered(env_cfg)
env.reset()

# Run some zero actions.
NUM_STEPS = 100
for i in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        if i > NUM_STEPS/2:
            microwave.open(env, env_ids=None)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

from isaac_arena.metrics.metrics import compute_metrics

print(f"Episode length buffer: {env.episode_length_buf}")
print(f"Max episode length: {env.max_episode_length}")
print(compute_metrics(env))



# %%
