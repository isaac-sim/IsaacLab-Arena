# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task definition for inserting a DisplayPort plug into a socket."""

from __future__ import annotations

from dataclasses import MISSING, dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.events import ResetRobotToObjectGraspPose
from isaaclab_arena.tasks.predicates.displayport_insertion import displayport_plug_is_inserted
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition
from isaaclab_arena.utils.cameras import ArenaCameraCfg, make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances


@dataclass(frozen=True)
class DisplayPortInsertionCriteria:
    """Thresholds that define a stable DisplayPort insertion."""

    plug_mating_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0221)
    """Mating point in the plug's local frame."""

    position_threshold: float = 0.005
    """Maximum mating-point position error in meters."""

    orientation_threshold_deg: float = 10.0
    """Maximum plug orientation error in degrees."""

    linear_velocity_threshold: float = 0.05
    """Maximum plug linear speed in meters per second."""

    angular_velocity_threshold: float = 0.5
    """Maximum plug angular speed in radians per second."""

    def __post_init__(self) -> None:
        assert len(self.plug_mating_offset_xyz) == 3, "plug_mating_offset_xyz must contain three values"
        assert self.position_threshold >= 0.0, "position_threshold must be non-negative"
        assert 0.0 <= self.orientation_threshold_deg <= 180.0, "orientation_threshold_deg must be in [0, 180]"
        assert self.linear_velocity_threshold >= 0.0, "linear_velocity_threshold must be non-negative"
        assert self.angular_velocity_threshold >= 0.0, "angular_velocity_threshold must be non-negative"


@dataclass(frozen=True)
class InitialObjectGraspCfg:
    """Robot-specific parameters used to initialize an object grasp."""

    robot_name: str
    arm_joint_names: tuple[str, ...]
    end_effector_body_name: str
    grasp_offset_xyz: tuple[float, float, float]
    gripper_close_command: dict[str, float]
    grasp_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    max_iterations: int = 150

    def __post_init__(self) -> None:
        assert self.robot_name, "robot_name must be set"
        assert self.arm_joint_names, "arm_joint_names must not be empty"
        assert self.end_effector_body_name, "end_effector_body_name must be set"
        assert len(self.grasp_offset_xyz) == 3, "grasp_offset_xyz must contain three values"
        assert len(self.grasp_rotation_xyzw) == 4, "grasp_rotation_xyzw must contain four values"
        assert self.gripper_close_command, "gripper_close_command must not be empty"
        assert self.max_iterations > 0, "max_iterations must be positive"


@register_task
class DisplayPortInsertionTask(TaskBase):
    """Insert a grasped DisplayPort plug into its fixed socket.

    Args:
        socket: Fixed DisplayPort socket asset.
        plug: DisplayPort plug manipulated by the robot.
        insertion_target: Socket-local frame representing the fully inserted plug pose.
        background_scene: Background whose minimum height defines a dropped plug.
        initial_grasp: Robot-specific reset configuration for the starting grasp.
        episode_length_s: Maximum episode duration in seconds.
        success_criteria: Geometric and motion thresholds for insertion.
        enable_cameras: Whether to add the task's workspace camera.
        task_description: Natural-language instruction. A description is generated when omitted.
    """

    def __init__(
        self,
        socket: Object,
        plug: Object,
        insertion_target: ObjectBase,
        background_scene: Asset,
        initial_grasp: InitialObjectGraspCfg,
        episode_length_s: float | None = None,
        success_criteria: DisplayPortInsertionCriteria | None = None,
        enable_cameras: bool = False,
        task_description: str | None = None,
    ) -> None:
        super().__init__(episode_length_s=episode_length_s)
        self.socket = socket
        self.plug = plug
        self.insertion_target = insertion_target
        self.background_scene = background_scene
        self.initial_grasp = initial_grasp
        self.success_criteria = success_criteria or DisplayPortInsertionCriteria()

        # A single full-scene reset restores the object states before the calibrated grasp is applied.
        self.socket.disable_reset_pose()
        self.plug.disable_reset_pose()
        self.events_cfg = DisplayPortInsertionEventsCfg(plug=plug, initial_grasp=initial_grasp)
        self.termination_cfg = self._make_termination_cfg()
        self.observation_cfg = DisplayPortInsertionObservationsCfg(socket=socket, plug=plug)
        self.camera_cfg = DisplayPortInsertionCameraCfg() if enable_cameras else None
        if self.camera_cfg is not None:
            self.observation_cfg = combine_configclass_instances(
                "DisplayPortInsertionObservationsCfg",
                self.observation_cfg,
                make_camera_observation_cfg(self.camera_cfg),
            )
        self.task_description = task_description or f"Insert the {plug.name} into the {socket.name}."

    def get_scene_cfg(self):
        return self.camera_cfg.get_cfg() if self.camera_cfg is not None else None

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.plug)]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(0.70, -0.45, 0.30), lookat=(0.475, 0.125, 0.0675))

    def _make_termination_cfg(self) -> DisplayPortInsertionTerminationsCfg:
        criteria = self.success_criteria
        success = TerminationTermCfg(
            func=displayport_plug_is_inserted,
            params={
                "plug_cfg": SceneEntityCfg(self.plug.name),
                "insertion_target": self.insertion_target,
                "plug_mating_offset_xyz": criteria.plug_mating_offset_xyz,
                "position_threshold": criteria.position_threshold,
                "orientation_threshold_deg": criteria.orientation_threshold_deg,
                "linear_velocity_threshold": criteria.linear_velocity_threshold,
                "angular_velocity_threshold": criteria.angular_velocity_threshold,
            },
        )
        plug_dropped = TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.plug.name),
            },
        )
        return DisplayPortInsertionTerminationsCfg(success=success, plug_dropped=plug_dropped)

    @classmethod
    def success_state_transition(cls, plug: str, socket: str, **_) -> TaskTransition:
        """Relate the inserted plug to its socket after success."""
        return TaskTransition(
            subject=plug,
            effects=(Relocate(subject=plug, relation="on", target=socket),),
        )


@configclass
class DisplayPortInsertionEventsCfg:
    """Reset terms for DisplayPort insertion."""

    reset_scene: EventTermCfg = MISSING
    initialize_grasp: EventTermCfg = MISSING

    def __init__(self, plug: Object, initial_grasp: InitialObjectGraspCfg) -> None:
        self.reset_scene = EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        self.initialize_grasp = EventTermCfg(
            func=ResetRobotToObjectGraspPose,
            mode="reset",
            params={
                "robot_cfg": SceneEntityCfg(initial_grasp.robot_name),
                "object_cfg": SceneEntityCfg(plug.name),
                "arm_joint_names": initial_grasp.arm_joint_names,
                "end_effector_body_name": initial_grasp.end_effector_body_name,
                "grasp_offset_xyz": initial_grasp.grasp_offset_xyz,
                "grasp_rotation_xyzw": initial_grasp.grasp_rotation_xyzw,
                "gripper_close_command": initial_grasp.gripper_close_command,
                "max_iterations": initial_grasp.max_iterations,
            },
        )


@configclass
class DisplayPortInsertionTerminationsCfg:
    """Termination terms for DisplayPort insertion."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    plug_dropped: TerminationTermCfg = MISSING


@configclass
class DisplayPortInsertionObservationsCfg:
    """Task-specific state observations for demonstration recording."""

    task_obs: ObsGroup = MISSING

    def __init__(self, socket: Object, plug: Object) -> None:
        @configclass
        class TaskObservationsCfg(ObsGroup):
            plug_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(plug.name)})
            plug_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(plug.name)})
            socket_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(socket.name)})
            socket_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(socket.name)})

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = False

        self.task_obs = TaskObservationsCfg()


@configclass
class DisplayPortInsertionCameraCfg(ArenaCameraCfg):
    """Fixed RGB camera overlooking the DisplayPort insertion workspace."""

    workspace_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/workspace_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.66,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.70, -0.45, 0.2975),
            rot=(0.56060, 0.10578, 0.15228, 0.80706),
            convention="opengl",
        ),
    )
