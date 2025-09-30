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
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaac_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()

destination_location = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
)
cracker_box.set_initial_pose(
    Pose(
        # Success
        position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
        # Fail
        # position_xyz=(0.0758066475391388 - 0.2, -0.5088448524475098 + 0.5, 0.0 + 0.2),
        rotation_wxyz=(1, 0, 0, 0),
    )
)

# %%


scene = Scene(assets=[background, cracker_box, destination_location])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
)

args_cli = get_isaac_arena_cli_parser().parse_args([])
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

from isaac_arena.metrics.metrics import compute_metrics

# Run some zero actions.
NUM_STEPS = 100
num_resets = 0
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        _, _, terminated, _, _ = env.step(actions)
        # print(torch.sum(terminated))
        num_resets += torch.sum(terminated).item()
        # print(f"terminated: {terminated}")
print(f"num_resets: {num_resets}")
print(compute_metrics(env))


# %%


import h5py
import pathlib

dataset_path = pathlib.Path("/tmp/isaaclab/logs/dataset.hdf5")

with h5py.File(dataset_path, "r") as f:
    print(f.keys())
    print(f["data"].keys())
    num_resets = len(f["data"])
    print(f"num_resets: {num_resets}")
    print(f["data"]["demo_2"].keys())
    print(f["data"]["demo_25"]["success"][:])
    print(f["data"]["demo_25"]["object_linear_velocity"][:])


# %%

# import pathlib
# import numpy as np
# import h5py

# # from isaac_arena.metrics.metric_base import MetricBase
# from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
# from isaaclab.utils.datasets import HDF5DatasetFileHandler


# # Extract recorded data for a metric
# def get_recorded_metric_data(dataset_path: pathlib.Path, recorder_term_name: str) -> list[np.ndarray]:
#     """Gets the recorded metric data for a given metric name."""
#     recorded_metric_data_per_demo: list[np.ndarray] = []
#     with h5py.File(dataset_path, "r") as f:
#         demos = f["data"]
#         for demo in demos:
#             recorded_metric_data_per_demo.append(demos[demo][recorder_term_name][:])
#     return recorded_metric_data_per_demo


# def get_metric_recorder_dataset_path(env: ManagerBasedRLEnv) -> pathlib.Path:
#     assert env.cfg.recorders.dataset_file_handler_class_type == HDF5DatasetFileHandler
#     return pathlib.Path(env.cfg.recorders.dataset_export_dir_path) / \
#                 pathlib.Path(env.cfg.recorders.dataset_filename + ".hdf5")


# # Compute all mertics
# # def compute_metrics(metrics: list[MetricBase], dataset_path: pathlib.Path) -> dict[str, float]:
# def compute_metrics(env: ManagerBasedRLEnv) -> dict[str, float]:
#     assert hasattr(env.cfg, "metrics")
#     # Get the path where the recorded data is stored
#     dataset_path = get_metric_recorder_dataset_path(env)
#     metrics_data = {}
#     for metric in env.cfg.metrics:
#         recorded_metric_data = get_recorded_metric_data(dataset_path, metric.recorder_term_name)
#         metrics_data[metric.name] = metric.compute_metric_from_recording(recorded_metric_data)
#     return metrics_data


# metrics_data = compute_metrics(env)
# print(f"Metrics data: {metrics_data}")

# %%


# recorded_success_rate_data = get_recorded_metric_data(dataset_path, "success")
# # print(f"Recorded metric data: {recorded_success_rate_data}")
# average_success_rate = success_metric.compute_metric_from_recording(recorded_success_rate_data)
# print(f"Average success rate: {average_success_rate}")

# recorded_object_linear_velocity_data = get_recorded_metric_data(dataset_path, "object_linear_velocity")
# # print(f"Recorded metric data: {recorded_object_linear_velocity_data}")
# average_object_moved_rate = object_moved_rate_metric.compute_metric_from_recording(recorded_object_linear_velocity_data)
# print(f"Average object moved rate: {average_object_moved_rate}")

# %%


# Calculate the average object velocity
# TODO: CHANGE FOR A THRESHOLD!


# recorded_metric_data = get_recorded_metric_data(dataset_path, "object_velocity")
# print(f"Recorded metric data: {recorded_metric_data}")

# OBJECT_VELOCITY_THRESHOLD = 0.5
# object_velocity_per_demo = recorded_metric_data
# object_moved_per_demo = []
# for object_velocity in object_velocity_per_demo:
#     print(f"Object velocity: {object_velocity}")
#     print(f"type of object_velocity: {type(object_velocity)}")
#     assert object_velocity.ndim == 2
#     assert object_velocity.shape[1] == 3
#     object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
#     print(f"Object linear velocity magnitude: {object_linear_velocity_magnitude}")
#     object_moved = np.any(object_linear_velocity_magnitude > OBJECT_VELOCITY_THRESHOLD)
#     print(f"Object moved: {object_moved}")
#     object_moved_per_demo.append(object_moved)
# object_moved_rate = np.mean(object_moved_per_demo)
# print(f"Object moved rate: {object_moved_rate}")

# #%%

# object_velocity = env.scene["cracker_box"].data.root_link_vel_w[:, :3]
# object_velocity_magnitude = torch.norm(object_velocity, dim=-1)
# print(f"Object velocity: {object_velocity}")
# print(f"Object velocity magnitude: {object_velocity_magnitude}")

# %%

# from isaaclab.managers import DatasetExportMode
# from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
# from isaaclab.utils import configclass


# class PrintingRecorder(RecorderTerm):

#     def record_post_step(self):
#         print("Recording post step")
#         return "printing", torch.zeros(self._env.num_envs, 1, device=self._env.device)


# class SuccessRecorder(RecorderTerm):

#     def record_pre_reset(self, env_ids):
#         assert hasattr(self._env, "termination_manager")
#         assert "success" in self._env.termination_manager.active_terms
#         success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
#         success_results |= self._env.termination_manager.get_term("success")[env_ids]
#         print(f"SuccessRecorder: {success_results}")
#         return "success", success_results


# @configclass
# class PrintingRecorderCfg(RecorderTermCfg):
#     class_type: type[RecorderTerm] = PrintingRecorder


# @configclass
# class PreResetPrintingRecorderCfg(RecorderTermCfg):
#     class_type: type[RecorderTerm] = PreResetPrintingRecorder


# @configclass
# class PrintingRecorderManagerCfg(RecorderManagerBaseCfg):

#     record_post_step_term = PrintingRecorderCfg()
#     record_pre_reset_term = PreResetPrintingRecorderCfg()


# recorder_cfg = PrintingRecorderManagerCfg()
# recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL


# %%
