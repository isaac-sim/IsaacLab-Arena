# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from .terminations import is_gripper_far


class G1FactoryPressButtonTask(TaskBase):
    """Factory-compatible button task preserving old gripper-near success semantics."""

    def __init__(
        self,
        pressable_object: Pressable,
        pressedness_threshold: float | None = None,
        reset_pressedness: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
        gripper_far_threshold: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(pressable_object, Pressable), "Pressable object must be an instance of Pressable"
        self.pressable_object = pressable_object
        self.pressedness_threshold = pressedness_threshold
        self.reset_pressedness = reset_pressedness
        self.robot_pose_range = robot_pose_range
        self.gripper_far_threshold = gripper_far_threshold
        self.task_description = (
            f"Press the {pressable_object.name} button" if task_description is None else task_description
        )

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        press_params = {}
        if self.pressedness_threshold is not None:
            press_params["pressedness_threshold"] = self.pressedness_threshold

        if self.gripper_far_threshold is not None:
            is_pressed = self.pressable_object.is_pressed
            object_name = self.pressable_object.name
            threshold = self.gripper_far_threshold

            def success_func(env):
                pressed = is_pressed(env, **press_params)
                gripper_near = ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(object_name),
                    threshold=threshold,
                )
                return pressed & gripper_near

            success = TerminationTermCfg(func=success_func, params={})
        else:
            success = TerminationTermCfg(func=self.pressable_object.is_pressed, params=press_params)
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return PressEventCfg(
            self.pressable_object,
            reset_pressedness=self.reset_pressedness,
            robot_pose_range=self.robot_pose_range,
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pressable_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class PressEventCfg:
    reset_button_state: EventTermCfg = MISSING
    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        pressable_object: Pressable,
        reset_pressedness: float | None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        assert isinstance(pressable_object, Pressable), "Object pose must be an instance of Pressable"
        params = {}
        if reset_pressedness is not None:
            params["unpressed_percentage"] = reset_pressedness
        self.reset_button_state = EventTermCfg(
            func=pressable_object.unpress,
            mode="reset",
            params=params,
        )
        self.reset_robot_pose = None
        if robot_pose_range is not None:
            self.reset_robot_pose = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": robot_pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
