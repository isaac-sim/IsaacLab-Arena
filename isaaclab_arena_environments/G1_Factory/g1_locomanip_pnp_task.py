# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
from isaaclab_arena.tasks.terminations import objects_in_proximity
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose, PoseRange

from .events import reset_all_distractors_uniform
from .terminations import is_gripper_far, is_in_contact, is_static, is_upright, object_near_fixed_position

# Default pose range is no randomization
_DEFAULT_POSE_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


def _is_physics_object(obj: Asset) -> bool:
    return isinstance(obj, ObjectBase) and obj.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION)


class G1LocomanipPnPTask(TaskBase):
    def __init__(
        self,
        object: Asset,
        destination: Asset,
        # lift_height: float = 0.15,
        drop_height: float | None = -0.6,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        max_x_separation: float = 0.1,
        max_y_separation: float = 0.1,
        max_z_separation: float = 0.859,
        tool: Asset | None = None,
        tool_max_separation: float = 0.1,
        distractors: list | None = None,
        require_static: bool = False,
        require_contact: bool = False,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        gripper_near_threshold: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        self.destination = destination
        # self.lift_height = lift_height
        self.drop_height = drop_height
        self.max_x_separation = max_x_separation
        self.max_y_separation = max_y_separation
        self.max_z_separation = max_z_separation
        self.tool = tool
        self.tool_max_separation = tool_max_separation
        self.distractors = distractors if distractors is not None else []
        self.require_static = require_static
        self.require_contact = require_contact
        self.upright_threshold = upright_threshold
        self.upright_z_axis_up = upright_z_axis_up
        self.gripper_far_threshold = gripper_far_threshold
        self.gripper_near_threshold = gripper_near_threshold
        self._contact_sensor_name: str | None = None
        self._contact_sensor_cfg = None
        if require_contact and _is_physics_object(object) and isinstance(destination, ObjectBase):
            self._contact_sensor_name = f"contact_sensor_{object.name}"
            self._contact_sensor_cfg = object.get_contact_sensor_cfg(contact_against_object=destination)
        self.task_description = (
            f"Pick up the {object.name}, and place it on the {destination.name}"
            if task_description is None
            else task_description
        )

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
            destination=self.destination,
            drop_height=self.drop_height,
            max_x_separation=self.max_x_separation,
            max_y_separation=self.max_y_separation,
            max_z_separation=self.max_z_separation,
            tool=self.tool,
            tool_max_separation=self.tool_max_separation,
            require_static=self.require_static,
            contact_sensor_name=self._contact_sensor_name,
            upright_threshold=self.upright_threshold,
            upright_z_axis_up=self.upright_z_axis_up,
            gripper_far_threshold=self.gripper_far_threshold,
            gripper_near_threshold=self.gripper_near_threshold,
        )

    def get_events_cfg(self):
        return EventCfg(object=self.object, tool=self.tool, distractors=self.distractors)

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
    tool_dropping: TerminationTermCfg | None = None
    success: TerminationTermCfg = MISSING

    def __init__(
        self,
        object: Asset,
        destination: Asset,
        drop_height: float | None = -0.05,
        max_x_separation: float = 0.1,
        max_y_separation: float = 0.1,
        max_z_separation: float = 0.859,
        tool: Asset | None = None,
        tool_max_separation: float = 0.1,
        require_static: bool = False,
        contact_sensor_name: str | None = None,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        gripper_near_threshold: float | None = None,
    ):
        self.object = object
        self.destination = destination
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.object_dropping = None
        self.tool_dropping = None
        self.gripper_near_threshold = gripper_near_threshold
        if drop_height is not None:
            if _is_physics_object(object):
                self.object_dropping = TerminationTermCfg(
                    func=mdp_isaac_lab.root_height_below_minimum,
                    params={"minimum_height": drop_height, "asset_cfg": SceneEntityCfg(object.name)},
                )
            if tool is not None and _is_physics_object(tool):
                self.tool_dropping = TerminationTermCfg(
                    func=mdp_isaac_lab.root_height_below_minimum,
                    params={"minimum_height": drop_height, "asset_cfg": SceneEntityCfg(tool.name)},
                )

        # Collect optional success criteria (each individually toggleable)
        extra_checks: list[tuple[callable, dict]] = []
        if _is_physics_object(object):
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
        if contact_sensor_name is not None:
            extra_checks.append((
                is_in_contact,
                {
                    "contact_sensor_cfg": SceneEntityCfg(contact_sensor_name),
                },
            ))

        # Gripper far: is_gripper_far — all fingers must be far from object
        # Maps to MuJoCo IsGripperFar(obj, threshold)
        if gripper_far_threshold is not None:
            extra_checks.append((
                is_gripper_far,
                {
                    "object_cfg": SceneEntityCfg(object.name),
                    "threshold": gripper_far_threshold,
                },
            ))

        # Gripper near: ~is_gripper_far — at least one finger must be close
        if gripper_near_threshold is not None:
            _near_th = gripper_near_threshold

            def _gripper_near(env):
                return ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(object.name),
                    threshold=_near_th,
                )

            extra_checks.append((_gripper_near, {}))

        # Determine base success function and params
        base_fn = None
        base_params: dict = {}

        if _is_physics_object(object) and _is_physics_object(destination):
            base_fn = objects_in_proximity
            base_params = {
                "object_cfg": SceneEntityCfg(self.object.name),
                "target_object_cfg": SceneEntityCfg(self.destination.name),
                "max_x_separation": max_x_separation,
                "max_y_separation": max_y_separation,
                "max_z_separation": max_z_separation,
            }
            if tool is not None:
                base_params["tool_cfg"] = SceneEntityCfg(tool.name)
                base_params["tool_max_x_separation"] = tool_max_separation
                base_params["tool_max_y_separation"] = tool_max_separation
                base_params["tool_max_z_separation"] = tool_max_separation
        elif _is_physics_object(object) and isinstance(destination, ObjectBase):
            dest_initial = destination.get_initial_pose()
            dest_pose = dest_initial.get_midpoint() if isinstance(dest_initial, PoseRange) else dest_initial
            if isinstance(dest_pose, Pose):
                base_fn = object_near_fixed_position
                base_params = {
                    "object_cfg": SceneEntityCfg(self.object.name),
                    "target_position": dest_pose.position_xyz,
                    "max_x_separation": max_x_separation,
                    "max_y_separation": max_y_separation,
                    "max_z_separation": max_z_separation,
                }

        # Build the success TerminationTermCfg, ANDing extra checks into it
        if base_fn is not None and extra_checks:
            _base_fn = base_fn
            _base_params = base_params
            _checks = extra_checks

            def _success(env):
                result = _base_fn(env, **_base_params)
                for check_fn, check_kw in _checks:
                    result = result & check_fn(env, **check_kw)
                return result

            self.success = TerminationTermCfg(func=_success, params={})
        elif base_fn is not None:
            self.success = TerminationTermCfg(func=base_fn, params=base_params)
        else:
            self.success = TerminationTermCfg(func=mdp_isaac_lab.time_out)


@configclass
class EventCfg:
    """Events for the MDP."""

    reset_all: EventTermCfg = MISSING
    reset_object_position: EventTermCfg | None = None
    reset_tool_position: EventTermCfg | None = None
    reset_distractors_position: EventTermCfg | None = None
    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        object: Asset,
        tool: Asset,
        distractors: list | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        self.reset_all = EventTermCfg(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")
        self.reset_object_position = None
        self.reset_tool_position = None
        self.reset_distractors_position = None
        self.reset_robot_pose = None
        if pose_range is None:
            pose_range = _DEFAULT_POSE_RANGE
        if robot_pose_range is None:
            robot_pose_range = _DEFAULT_POSE_RANGE

        has_own_randomization = isinstance(object, ObjectBase) and (
            isinstance(object.get_initial_pose(), PoseRange)
            or (object.event_cfg is not None and object.event_cfg.func is not set_object_pose)
        )
        if _is_physics_object(object) and not has_own_randomization:
            self.reset_object_position = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            )

        if tool is not None and _is_physics_object(tool):
            self.reset_tool_position = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(tool.name),
                },
            )

        if distractors is not None and len(distractors) > 0:
            distractor_names = [d.name for d in distractors]
            self.reset_distractors_position = EventTermCfg(
                func=reset_all_distractors_uniform,
                mode="reset",
                params={
                    "distractor_names": distractor_names,
                    "pose_range": pose_range,
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
