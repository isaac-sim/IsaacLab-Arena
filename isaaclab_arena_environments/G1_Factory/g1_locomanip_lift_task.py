# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import PoseRange

from .terminations import is_grasped, is_gripper_far, is_in_contact, is_static, is_upright, object_lifted

# Default pose range is no randomization
_DEFAULT_POSE_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


class G1LocomanipLiftTask(TaskBase):
    def __init__(
        self,
        object: Asset,
        lift_height: float = 0.15,
        drop_height: float | None = None,
        episode_length_s: float | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        task_description: str | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
        require_static: bool = False,
        contact_sensor_name: str | None = None,
        contact_force_threshold: float = 1.0,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        require_grasped: bool = False,
        not_in_contact_with: Asset | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        self.lift_height = lift_height
        self.drop_height = drop_height
        self.pose_range = pose_range if pose_range is not None else _DEFAULT_POSE_RANGE
        self.task_description = task_description
        self.robot_pose_range = robot_pose_range
        self.require_static = require_static
        self.contact_sensor_name = contact_sensor_name
        self.contact_force_threshold = contact_force_threshold
        self.upright_threshold = upright_threshold
        self.upright_z_axis_up = upright_z_axis_up
        self.gripper_far_threshold = gripper_far_threshold
        self.require_grasped = require_grasped
        self.not_in_contact_with = not_in_contact_with

        self._contact_sensor_name: str | None = None
        self._contact_sensor_cfg = None
        if (
            not_in_contact_with is not None
            and isinstance(object, ObjectBase)
            and isinstance(not_in_contact_with, ObjectBase)
        ):
            self._contact_sensor_name = f"contact_sensor_{object.name}"
            self._contact_sensor_cfg = object.get_contact_sensor_cfg(contact_against_object=not_in_contact_with)

    def get_scene_cfg(self):
        if self._contact_sensor_cfg is not None:
            SceneCfg = make_configclass(
                "SceneCfg",
                [(self._contact_sensor_name, type(self._contact_sensor_cfg), self._contact_sensor_cfg)],
            )
            return SceneCfg()
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(
            object=self.object,
            lift_height=self.lift_height,
            drop_height=self.drop_height,
            require_static=self.require_static,
            contact_sensor_name=self.contact_sensor_name,
            contact_force_threshold=self.contact_force_threshold,
            upright_threshold=self.upright_threshold,
            upright_z_axis_up=self.upright_z_axis_up,
            gripper_far_threshold=self.gripper_far_threshold,
            require_grasped=self.require_grasped,
            not_in_contact_sensor_name=self._contact_sensor_name,
        )

    def get_events_cfg(self):
        return EventCfg(object=self.object, pose_range=self.pose_range, robot_pose_range=self.robot_pose_range)

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.object,
            offset=np.array([-1.3, 1.7, 1.5]),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = MISSING
    object_dropping: TerminationTermCfg | None = None
    success: TerminationTermCfg = MISSING

    def __init__(
        self,
        object: Asset,
        lift_height: float = 0.15,
        drop_height: float | None = -0.05,
        require_static: bool = False,
        contact_sensor_name: str | None = None,
        contact_force_threshold: float = 1.0,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        require_grasped: bool = False,
        not_in_contact_sensor_name: str | None = None,
    ):
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.object_dropping = None
        self.success = TerminationTermCfg(func=mdp_isaac_lab.time_out)

        is_physics = isinstance(object, ObjectBase) and object.object_type in (
            ObjectType.RIGID,
            ObjectType.ARTICULATION,
        )
        if not is_physics:
            return

        if drop_height is not None:
            self.object_dropping = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": drop_height, "asset_cfg": SceneEntityCfg(object.name)},
            )

        # Collect optional success criteria (each individually toggleable)
        extra_checks: list[tuple[callable, dict]] = []
        if require_static:
            extra_checks.append((
                is_static,
                {
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            ))
        if upright_threshold is not None:
            extra_checks.append((
                is_upright,
                {
                    "asset_cfg": SceneEntityCfg(object.name),
                    "threshold": upright_threshold,
                    "z_axis_up": upright_z_axis_up,
                },
            ))

        # For lift tasks, gripper_far_threshold is NEGATED: success requires
        # the gripper to be NEAR the object (robot is still holding it).
        # Maps to MuJoCo NotCriteria(IsGripperFar(obj)).
        if gripper_far_threshold is not None:
            _th = gripper_far_threshold

            def _gripper_near(env):
                return ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(object.name),
                    threshold=_th,
                )

            extra_checks.append((_gripper_near, {}))

        # Grasped: at least one finger is close to the object.
        # Maps to MuJoCo IsGrasped(obj).
        if require_grasped:
            extra_checks.append((
                is_grasped,
                {
                    "object_cfg": SceneEntityCfg(object.name),
                },
            ))

        if not_in_contact_sensor_name is not None:
            _sensor_name = not_in_contact_sensor_name

            def _not_in_contact(env):
                return ~is_in_contact(
                    env,
                    contact_sensor_cfg=SceneEntityCfg(_sensor_name),
                )

            extra_checks.append((_not_in_contact, {}))

        base_params = {"minimum_height": lift_height, "asset_cfg": SceneEntityCfg(object.name)}

        if extra_checks:
            _base_params = base_params
            _checks = extra_checks

            def _success(env):
                result = object_lifted(env, **_base_params)
                for check_fn, check_kw in _checks:
                    result = result & check_fn(env, **check_kw)
                return result

            self.success = TerminationTermCfg(func=_success, params={})
        else:
            self.success = TerminationTermCfg(func=object_lifted, params=base_params)


@configclass
class EventCfg:
    """Events for the MDP."""

    reset_all: EventTermCfg = MISSING
    reset_object_position: EventTermCfg | None = None
    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        object: Asset,
        pose_range: dict[str, tuple[float, float]] | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        if pose_range is None:
            pose_range = _DEFAULT_POSE_RANGE
        if robot_pose_range is None:
            robot_pose_range = _DEFAULT_POSE_RANGE

        self.reset_all = EventTermCfg(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")
        self.reset_object_position = None
        self.reset_robot_pose = None

        is_physics_object = isinstance(object, ObjectBase) and object.object_type in (
            ObjectType.RIGID,
            ObjectType.ARTICULATION,
        )
        # Skip if the object already has its own randomization:
        #   - PoseRange initial pose (randomizes via randomize_object_pose)
        #   - Manually-set event_cfg with a custom function (e.g. set_object_pose_random_choice)
        # Only add task randomization when the object has no event or only the
        # auto-generated fixed-pose reset (set_object_pose).
        has_own_randomization = isinstance(object, ObjectBase) and (
            isinstance(object.get_initial_pose(), PoseRange)
            or (object.event_cfg is not None and object.event_cfg.func is not set_object_pose)
        )
        if is_physics_object and not has_own_randomization:
            self.reset_object_position = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            )

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
