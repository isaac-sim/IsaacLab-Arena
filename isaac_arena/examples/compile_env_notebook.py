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
from isaac_arena.assets.object_reference import ObjectReference
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.dummy_task import DummyTask
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)
cracker_box.set_initial_pose(
    Pose(
        position_xyz=(0.0758066475391388 + 0.2, -0.5088448524475098, 0.0 + 0.2),
        rotation_wxyz=(1, 0, 0, 0),
    )
)


# %%

# from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
# from isaaclab.managers import DatasetExportMode

# recorder_cfg = ActionStateRecorderManagerCfg()
# recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE

# %%

from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PrintingRecorder(RecorderTerm):

    def record_post_step(self):
        print("Recording post step")
        return "printing", torch.zeros(self._env.num_envs, 1, device=self._env.device)


class PreResetPrintingRecorder(RecorderTerm):

    def record_pre_reset(self, env_ids):
        print("Recording pre reset")
        return "printing pre reset", torch.zeros(self._env.num_envs, 1, device=self._env.device)


@configclass
class PrintingRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PrintingRecorder


@configclass
class PreResetPrintingRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreResetPrintingRecorder


@configclass
class PrintingRecorderManagerCfg(RecorderManagerBaseCfg):

    record_post_step_term = PrintingRecorderCfg()
    record_pre_reset_term = PreResetPrintingRecorderCfg()


recorder_cfg = PrintingRecorderManagerCfg()
recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE


# %%

# # Set task success values for the relevant episodes
# success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
# # Check success indicator from termination terms
# asset hasattr(self._env, "termination_manager"):
# if "success" in self._env.termination_manager.active_terms:
#         success_results |= self._env.termination_manager.get_term("success")[env_ids]


# %%


scene = Scene(assets=[background, cracker_box, destination_location])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    # task=DummyTask(),
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
    recorder_cfg=recorder_cfg,
)

args_cli = get_isaac_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()


# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%

for i in range(10):
    print(i)
    print(env.recorder_manager.get_episode(i).data["printing"].shape)

# %%
