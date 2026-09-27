# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from functools import partial

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.placeable import Placeable
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class PlaceUprightTask(TaskBase):

    def __init__(
        self,
        placeable_object: Placeable,
        orientation_threshold: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(placeable_object, Placeable), "Placeable object must be an instance of Placeable"
        self.placeable_object = placeable_object
        self.orientation_threshold = (
            orientation_threshold if orientation_threshold is not None else placeable_object.orientation_threshold
        )
        self.scene_config = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.events_cfg = None
        self.task_description = (
            f"Place the {placeable_object.name} upright" if task_description is None else task_description
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self) -> TaskTerminationCfg:
        params = {}
        if self.orientation_threshold is not None:
            params["orientation_threshold"] = self.orientation_threshold
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="place_upright",
                    predicate_sequence=[partial(self.placeable_object.is_placed_upright, **params)],
                )
            ],
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return PlaceUprightMimicEnvCfg(
            arm_mode=arm_mode,
            placeable_object_name=self.placeable_object.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            ObjectMovedRateMetric(self.placeable_object),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.placeable_object, offset=np.array([1.5, 1.5, 1.5]))


@configclass
class PlaceUprightMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Place Upright env.
    """

    arm_mode: ArmMode = ArmMode.SINGLE_ARM

    placeable_object_name: str = "placeable_object"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_placeupright_isaac_lab_task_D0"
        # Use default mimic datagen config parameters
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.placeable_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(0, 0),
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
                object_ref=self.placeable_object_name,
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
        if self.arm_mode == ArmMode.SINGLE_ARM:
            self.subtask_configs["robot"] = subtask_configs
        # We need to add the left and right subtasks for GR1.
        elif self.arm_mode in [ArmMode.LEFT, ArmMode.RIGHT]:
            self.subtask_configs[self.arm_mode.value] = subtask_configs
            # EEF on opposite side (arm is static)
            subtask_configs = []
            subtask_configs.append(
                SubTaskConfig(
                    # Each subtask involves manipulation with respect to a single object frame.
                    object_ref=self.placeable_object_name,
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
            self.subtask_configs[self.arm_mode.get_other_arm().value] = subtask_configs

        else:
            raise ValueError(f"Embodiment arm mode {self.arm_mode} not supported")
