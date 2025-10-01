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

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaac_arena.affordances.openable import Openable
from isaac_arena.metrics.metric_base import MetricBase
from isaac_arena.metrics.success_rate import SuccessRateMetric
from isaac_arena.tasks.task_base import TaskBase
from isaac_arena.terms.events import set_object_pose


class OpenDoorTask(TaskBase):
    def __init__(
        self, openable_object: Openable, openness_threshold: float | None = None, reset_openness: float | None = None
    ):
        super().__init__()
        assert isinstance(openable_object, Openable), "Openable object must be an instance of Openable"
        self.openable_object = openable_object
        self.openness_threshold = openness_threshold
        self.reset_openness = reset_openness
        self.scene_config = None
        self.events_cfg = OpenDoorEventCfg(self.openable_object, reset_openness=self.reset_openness)
        self.termination_cfg = self.make_termination_cfg()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        params = {}
        if self.openness_threshold is not None:
            params["threshold"] = self.openness_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_open,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        return OpenDoorMimicEnvCfg(
            embodiment_name=embodiment_name,
            openable_object_name=self.openable_object.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        # TODO(alexmillane, 2025.09.25): Add door moved metric.
        return [SuccessRateMetric()]


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=False)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING


@configclass
class OpenDoorEventCfg:
    """Configuration for Open Door."""

    reset_door_state: EventTermCfg = MISSING

    reset_openable_object_pose: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openness: float | None):
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openness is not None:
            params["percentage"] = reset_openness
        self.reset_door_state = EventTermCfg(
            func=openable_object.close,
            mode="reset",
            params=params,
        )
        initial_pose = openable_object.get_initial_pose()
        if initial_pose is not None:
            self.reset_openable_object_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(openable_object.name),
                },
            )


@configclass
class OpenDoorMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Open Door env.
    """

    embodiment_name: str = "franka"

    openable_object_name: str = "openable_object"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_opendoor_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.openable_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                # TODO(alexmillane, 2025.09.02): This is currently broken. FIX.
                # We need a way to pass in a reference to an object that exists in the
                # scene.
                object_ref=self.openable_object_name,
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        if self.embodiment_name == "franka":
            self.subtask_configs["robot"] = subtask_configs
        # We need to add the left and right subtasks for GR1.
        elif self.embodiment_name == "gr1":
            self.subtask_configs["right"] = subtask_configs
            # EEF on opposite side (arm is static)
            subtask_configs = []
            subtask_configs.append(
                SubTaskConfig(
                    # Each subtask involves manipulation with respect to a single object frame.
                    object_ref=self.openable_object_name,
                    # Corresponding key for the binary indicator in "datagen_info" for completion
                    subtask_term_signal=None,
                    # Time offsets for data generation when splitting a trajectory
                    subtask_term_offset_range=(0, 0),
                    # Selection strategy for source subtask segment
                    selection_strategy="nearest_neighbor_object",
                    # Optional parameters for the selection strategy function
                    selection_strategy_kwargs={"nn_k": 3},
                    # Amount of action noise to apply during this subtask
                    action_noise=0.005,
                    # Number of interpolation steps to bridge to this subtask segment
                    num_interpolation_steps=0,
                    # Additional fixed steps for the robot to reach the necessary pose
                    num_fixed_steps=0,
                    # If True, apply action noise during the interpolation phase and execution
                    apply_noise_during_interpolation=False,
                )
            )
            self.subtask_configs["left"] = subtask_configs

        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
