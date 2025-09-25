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
        # Success
        position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
        # Fail
        # position_xyz=(0.0758066475391388 - 0.2, -0.5088448524475098 + 0.5, 0.0 + 0.2),
        rotation_wxyz=(1, 0, 0, 0),
    )
)


# %%

import numpy as np

from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaac_arena.metrics.metric_base import MetricBase


class SuccessRecorder(RecorderTerm):

    name = "success"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # We track the first reset for each environment
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        # The first time that the environment is reset, we don't want to record the success,
        # because nothing has happened yet.
        if self.first_reset:
            # We expect that on the first reset ALL the environments are reset.
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            # Record nothing.
            return None, None
        assert hasattr(self._env, "termination_manager")
        assert "success" in self._env.termination_manager.active_terms
        success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
        success_results |= self._env.termination_manager.get_term("success")[env_ids]
        print(f"SuccessRecorder. env_ids: {env_ids}. success_results: {success_results}")
        return self.name, success_results


class SuccessRateMetric(MetricBase):

    name = "success"
    recorder_term_name = SuccessRecorder.name

    def get_recorder_term_cfg(self):
        return SuccessRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        """Gets the average success rate from a list of recorded success flags."""
        num_demos = len(recorded_metric_data)
        all_demos_success_flags = np.concatenate(recorded_metric_data)
        assert all_demos_success_flags.ndim == 1
        assert all_demos_success_flags.shape[0] == num_demos
        success_rate = np.mean(all_demos_success_flags)
        return success_rate


@configclass
class SuccessRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SuccessRecorder


class ObjectVelocityRecorder(RecorderTerm):

    name = "object_linear_velocity"

    def record_post_step(self):
        object_linear_velocity = self._env.scene["cracker_box"].data.root_link_vel_w[:, :3]
        assert object_linear_velocity.shape == (self._env.num_envs, 3)
        return self.name, object_linear_velocity


class ObjectMovedRateMetric(MetricBase):

    name = "object_moved"
    recorder_term_name = ObjectVelocityRecorder.name

    def __init__(self, object_velocity_threshold: float = 0.5):
        super().__init__()
        self.object_velocity_threshold = object_velocity_threshold

    def get_recorder_term_cfg(self):
        return ObjectVelocityRecorderCfg()

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        object_velocity_per_demo = recorded_metric_data
        object_moved_per_demo = []
        for object_velocity in object_velocity_per_demo:
            assert object_velocity.ndim == 2
            assert object_velocity.shape[1] == 3
            object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
            object_moved = np.any(object_linear_velocity_magnitude > self.object_velocity_threshold)
            object_moved_per_demo.append(object_moved)
        object_moved_rate = np.mean(object_moved_per_demo)
        return object_moved_rate


@configclass
class ObjectVelocityRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObjectVelocityRecorder


success_metric = SuccessRateMetric()
object_moved_rate_metric = ObjectMovedRateMetric()

# metrics: dict[str, MetricBase] = {
#     success_metric.name: success_metric,
#     object_moved_rate_metric.name: object_moved_rate_metric,
# }
metrics: list[MetricBase] = [success_metric, object_moved_rate_metric]

from isaac_arena.utils.configclass import make_configclass


def metrics_to_recorder_manager_cfg(metrics: list[MetricBase]) -> RecorderManagerBaseCfg:
    configclass_fields: list[tuple[str, type, object]] = []
    for metric in metrics:
        configclass_fields.append((metric.name, type(metric.get_recorder_term_cfg()), metric.get_recorder_term_cfg()))
    recorder_cfg_cls = make_configclass("RecorderManagerCfg", configclass_fields, bases=(RecorderManagerBaseCfg,))
    recorder_cfg = recorder_cfg_cls()
    recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    return recorder_cfg


recorder_manager_cfg = metrics_to_recorder_manager_cfg(metrics)

# %%


scene = Scene(assets=[background, cracker_box, destination_location])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    # task=DummyTask(),
    task=PickAndPlaceTask(cracker_box, destination_location, background),
    teleop_device=None,
    recorder_cfg=recorder_manager_cfg,
)

args_cli = get_isaac_arena_cli_parser().parse_args([])
args_cli.num_envs = 2
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()


# %%

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


# %%


import h5py
import pathlib

dataset_path = pathlib.Path("/tmp/isaaclab/logs/dataset.hdf5")

with h5py.File(dataset_path, "r") as f:
    print(f.keys())
    print(f["data"].keys())
    # print(f["data"]["demo_2"].keys())
    print(f["data"]["demo_1"]["success"][:])
    print(f["data"]["demo_1"]["object_linear_velocity"][:])


# %%

import numpy as np


# Extract recorded data for a metric
def get_recorded_metric_data(dataset_path: pathlib.Path, recorder_term_name: str) -> list[np.ndarray]:
    """Gets the recorded metric data for a given metric name."""
    recorded_metric_data_per_demo: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        for demo in demos:
            print(f"demo data: {demos[demo].keys()}")
            print(f"metric name: {recorder_term_name}")
            recorded_metric_data_per_demo.append(demos[demo][recorder_term_name][:])
    return recorded_metric_data_per_demo


# Compute all mertics
def compute_metrics(metrics: list[MetricBase], dataset_path: pathlib.Path) -> dict[str, float]:
    metrics_data = {}
    for metric in metrics:
        recorded_metric_data = get_recorded_metric_data(dataset_path, metric.recorder_term_name)
        metrics_data[metric.name] = metric.compute_metric_from_recording(recorded_metric_data)
    return metrics_data


metrics_data = compute_metrics(metrics, dataset_path)
print(f"Metrics data: {metrics_data}")

# %%


recorded_success_rate_data = get_recorded_metric_data(dataset_path, "success")
# print(f"Recorded metric data: {recorded_success_rate_data}")
average_success_rate = success_metric.compute_metric_from_recording(recorded_success_rate_data)
print(f"Average success rate: {average_success_rate}")

recorded_object_linear_velocity_data = get_recorded_metric_data(dataset_path, "object_linear_velocity")
# print(f"Recorded metric data: {recorded_object_linear_velocity_data}")
average_object_moved_rate = object_moved_rate_metric.compute_metric_from_recording(recorded_object_linear_velocity_data)
print(f"Average object moved rate: {average_object_moved_rate}")

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
