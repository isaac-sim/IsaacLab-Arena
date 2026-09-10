# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual I2RT YAM embodiment used by Isaac Cap cable routing."""

from __future__ import annotations

import math
import torch
from collections.abc import Mapping, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
END_EFFECTOR_BODY_NAME = "link_6"
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0

_DEFAULT_ARM_JOINT_POSITIONS = (0.0, 0.85, 0.60, 0.0, 0.0, 0.0)
_CAMERA_WIDTH = 1280
_CAMERA_HEIGHT = 720
_D405_MOUNT_POSITION = (-0.0107, 0.079729, 0.066021)
_D405_MOUNT_ROTATION_XYZW = (0.423, 0.0, 0.0, 0.906)
_TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT = (0.335, 0.0, 0.93732053)
_TOP_CAMERA_ROTATION_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


class FiniteJointPositionAction(JointPositionAction):
    """Keep absolute joint targets finite and inside the soft limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize and constrain absolute joint-position commands."""
        finite_actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        super().process_actions(finite_actions)
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.where(torch.isfinite(self._processed_actions), self._processed_actions, default)
        self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


class NormalizedFiniteJointPositionAction(FiniteJointPositionAction):
    """Map a finite command in ``[0, 1]`` through the joint transform."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clamp the normalized command before applying its configured transform."""
        normalized_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        super().process_actions(normalized_actions)


@configclass
class FiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure finite absolute joint-position actions."""

    class_type: type[FiniteJointPositionAction] = FiniteJointPositionAction


@configclass
class NormalizedFiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure normalized finite joint-position actions."""

    class_type: type[NormalizedFiniteJointPositionAction] = NormalizedFiniteJointPositionAction


def _yam_articulation(
    prim_path: str,
    position: tuple[float, float, float],
    robot_usd_path: str,
) -> ArticulationCfg:
    joint_pos = dict(zip(ARM_JOINT_NAMES, _DEFAULT_ARM_JOINT_POSITIONS, strict=True))
    joint_pos[GRIPPER_JOINT_NAME] = GRIPPER_OPEN_POSITION
    joint_pos[PASSIVE_GRIPPER_JOINT_NAME] = -GRIPPER_OPEN_POSITION
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd_path),
            copy_from_source=False,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "arm_joints_1_3": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                stiffness=80.0,
                damping=6.0,
                joint_effort_limit=28.0,
            ),
            "arm_joint_4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                stiffness=30.0,
                damping=2.0,
                joint_effort_limit=10.0,
            ),
            "arm_joints_5_6": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"],
                stiffness=30.0,
                damping=2.0,
                joint_effort_limit=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_NAME],
                stiffness=1000.0,
                damping=100.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[PASSIVE_GRIPPER_JOINT_NAME],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


def _arm_action(asset_name: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


def _gripper_action(asset_name: str) -> NormalizedFiniteJointPositionActionCfg:
    return NormalizedFiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=[GRIPPER_JOINT_NAME],
        scale=GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION,
        offset=GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class BimanualYamSceneCfg:
    """Two independently addressable YAM articulations."""

    left_robot: ArticulationCfg | None = None
    right_robot: ArticulationCfg | None = None


@configclass
class BimanualYamActionsCfg:
    """Left arm/gripper followed by right arm/gripper in policy order."""

    left_arm_action: FiniteJointPositionActionCfg = _arm_action("left_robot")
    left_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("left_robot")
    right_arm_action: FiniteJointPositionActionCfg = _arm_action("right_robot")
    right_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("right_robot")


def _robot(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _index(names: list[str], expected: str, kind: str, side: str) -> int:
    try:
        return names.index(expected)
    except ValueError as error:
        raise ValueError(f"{side} is missing required {kind} {expected!r}") from error


def arm_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return arm positions in declared action order for one YAM."""
    robot = _robot(env, asset_cfg)
    indices = [_index(robot.data.joint_names, name, "joint", asset_cfg.name) for name in ARM_JOINT_NAMES]
    return robot.data.joint_pos.torch[:, indices]


def gripper_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return zero-open, one-closed state for one YAM."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.joint_names, GRIPPER_JOINT_NAME, "joint", asset_cfg.name)
    position = robot.data.joint_pos.torch[:, index : index + 1]
    travel = GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION
    return torch.clamp((position - GRIPPER_OPEN_POSITION) / travel, min=0.0, max=1.0)


def ee_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one YAM's end-effector world position."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name)
    return robot.data.body_pos_w.torch[:, index, :]


def ee_quat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one YAM's end-effector world quaternion."""
    robot = _robot(env, asset_cfg)
    index = _index(robot.data.body_names, END_EFFECTOR_BODY_NAME, "body", asset_cfg.name)
    return robot.data.body_quat_w.torch[:, index, :]


@configclass
class BimanualYamObservationsCfg:
    """Ordered, side-qualified policy observations for both YAMs."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            for side in ("left", "right"):
                asset_cfg = SceneEntityCfg(f"{side}_robot")
                setattr(self, f"{side}_joint_pos", ObsTerm(func=arm_joint_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_gripper_pos", ObsTerm(func=gripper_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_eef_pos", ObsTerm(func=ee_pos, params={"asset_cfg": asset_cfg}))
                setattr(self, f"{side}_eef_quat", ObsTerm(func=ee_quat, params={"asset_cfg": asset_cfg}))
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def _d405_camera(
    prim_path: str,
    *,
    position: tuple[float, float, float] = _D405_MOUNT_POSITION,
    rotation_xyzw: tuple[float, float, float, float] = _D405_MOUNT_ROTATION_XYZW,
) -> CameraCfg:
    vertical_aperture = 4.8
    vertical_fov_deg = 58.0
    focal_length = vertical_aperture / (2.0 * math.tan(math.radians(vertical_fov_deg / 2.0)))
    return CameraCfg(
        prim_path=prim_path,
        height=_CAMERA_HEIGHT,
        width=_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=6.4,
            vertical_aperture=vertical_aperture,
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation_xyzw, convention="ros"),
    )


@configclass
class BimanualYamCameraCfg(ArenaCameraCfg):
    """One centered top camera and one D405 below each YAM end effector."""

    left_wrist_camera: CameraCfg = _d405_camera(f"{{ENV_REGEX_NS}}/LeftRobot{_LINK_SIX_SUFFIX}/left_wrist_camera")
    right_wrist_camera: CameraCfg = _d405_camera(f"{{ENV_REGEX_NS}}/RightRobot{_LINK_SIX_SUFFIX}/right_wrist_camera")
    top_camera: CameraCfg = _d405_camera(
        "{ENV_REGEX_NS}/top_camera",
        position=_TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT,
        rotation_xyzw=_TOP_CAMERA_ROTATION_XYZW,
    )

    def set_robot_mount_positions(
        self,
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> None:
        """Place the top camera at Cap's offset from the robot midpoint."""
        midpoint = tuple((float(a) + float(b)) * 0.5 for a, b in zip(left, right, strict=True))
        self.top_camera.offset.pos = tuple(
            midpoint_axis + offset_axis
            for midpoint_axis, offset_axis in zip(midpoint, _TOP_CAMERA_OFFSET_FROM_ROBOT_MIDPOINT, strict=True)
        )


class IndustrialBimanualYamEmbodiment(EmbodimentBase):
    """Two fixed-base YAM manipulators controlled by absolute joint targets."""

    name = "industrial_bimanual_yam"
    tags = ["embodiment", "yam", "bimanual"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        *,
        robot_usd_path: str,
        instanceable_robot_usd_path: str,
        left_mount_position: Sequence[float],
        right_mount_position: Sequence[float],
        enable_cameras: bool = False,
        use_tiled_cameras: bool = False,
        use_instanceable_meshes: bool = False,
    ) -> None:
        """Configure the fixed cable-routing YAM pair."""
        left_position = tuple(float(value) for value in left_mount_position)
        right_position = tuple(float(value) for value in right_mount_position)
        assert len(left_position) == len(right_position) == 3, "YAM mount positions must contain three values."
        self._robot_usd_path = robot_usd_path
        self._instanceable_robot_usd_path = instanceable_robot_usd_path
        active_usd_path = instanceable_robot_usd_path if use_instanceable_meshes else robot_usd_path

        super().__init__(
            enable_cameras=enable_cameras,
            concatenate_observation_terms=True,
            arm_mode=ArmMode.DUAL_ARM,
        )
        self.scene_config = BimanualYamSceneCfg(
            left_robot=_yam_articulation("{ENV_REGEX_NS}/LeftRobot", left_position, active_usd_path),
            right_robot=_yam_articulation("{ENV_REGEX_NS}/RightRobot", right_position, active_usd_path),
        )
        self.action_config = BimanualYamActionsCfg()
        self.observation_config = BimanualYamObservationsCfg()
        self.camera_config = BimanualYamCameraCfg() if enable_cameras else None
        if self.camera_config is not None:
            self.camera_config.set_use_tiled_camera(use_tiled_cameras)
            self.camera_config.set_robot_mount_positions(left_position, right_position)
            self.add_camera_variations(self.camera_config)

    def get_scene_key(self) -> str:
        """Return the left articulation as the primary scene key."""
        return "left_robot"

    def get_initial_pose(self) -> Pose:
        """Return the midpoint pose of the fixed bimanual layout."""
        left = self.scene_config.left_robot.init_state.pos
        right = self.scene_config.right_robot.init_state.pos
        midpoint = tuple((float(a) + float(b)) * 0.5 for a, b in zip(left, right, strict=True))
        return Pose(position_xyz=midpoint)

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update both YAM articulations' initial joint positions."""
        self.scene_config.left_robot.init_state.joint_pos.update(joint_pos)
        self.scene_config.right_robot.init_state.joint_pos.update(joint_pos)

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME
