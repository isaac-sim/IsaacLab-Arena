# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-base bimanual YAM embodiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import JointDriveBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.sim.schemas import (
    MujocoRigidBodyPropertiesCfg,
    NewtonArticulationRootPropertiesCfg,
    NewtonCollisionPropertiesCfg,
)

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.yam.actions import FiniteJointPositionActionCfg, NormalizedFiniteJointPositionActionCfg
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

YAM_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
YAM_USD_PATH = YAM_ASSET_DIRECTORY / "i2rt_yam_cable_routing.usda"

YAM_GRIPPER_OPEN_POSITION = 0.0375
YAM_GRIPPER_CLOSED_POSITION = 0.0
YAM_CONTACT_GAP = 0.001
YAM_DEFAULT_LEFT_POSITION = (-0.335, 0.30, 0.767)
YAM_DEFAULT_RIGHT_POSITION = (-0.335, -0.30, 0.767)
YAM_DEFAULT_JOINT_POSITIONS = {
    "joint1": 0.0,
    "joint2": 0.85,
    "joint3": 0.60,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "left_finger": YAM_GRIPPER_OPEN_POSITION,
    "right_finger": -YAM_GRIPPER_OPEN_POSITION,
}


def make_yam_articulation_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    yaw: float = 0.0,
) -> ArticulationCfg:
    """Create one fixed-base YAM articulation configured for Newton."""
    assert YAM_USD_PATH.is_file(), f"YAM asset not found: {YAM_USD_PATH}"
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD_PATH),
            copy_from_source=False,
            rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=False),
            collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=YAM_CONTACT_GAP),
            joint_drive_props=JointDriveBaseCfg(ensure_drives_exist=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
            joint_pos=YAM_DEFAULT_JOINT_POSITIONS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                joint_effort_limit=28.0,
                stiffness=80.0,
                damping=6.0,
            ),
            "arm_joint_4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                joint_effort_limit=10.0,
                stiffness=30.0,
                damping=2.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"],
                joint_effort_limit=10.0,
                stiffness=30.0,
                damping=2.0,
            ),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["left_finger"],
                stiffness=1000.0,
                damping=100.0,
            ),
            # MJWarp imports the source YAM's -1 equality from left_finger to
            # right_finger, so the mirrored coordinate remains passive.
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=["right_finger"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def _arm_action_cfg(asset_name: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=["joint[1-6]"],
        preserve_order=True,
        use_default_offset=False,
    )


def _gripper_action_cfg(asset_name: str) -> NormalizedFiniteJointPositionActionCfg:
    return NormalizedFiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=["left_finger"],
        scale=YAM_GRIPPER_CLOSED_POSITION - YAM_GRIPPER_OPEN_POSITION,
        offset=YAM_GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class BimanualYamSceneCfg:
    """The left and right YAM articulations."""

    left_robot: ArticulationCfg | None = None
    right_robot: ArticulationCfg | None = None


@configclass
class BimanualYamActionsCfg:
    """Four ordered action terms forming a 14-dimensional action space."""

    left_arm_action: FiniteJointPositionActionCfg = _arm_action_cfg("left_robot")
    left_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action_cfg("left_robot")
    right_arm_action: FiniteJointPositionActionCfg = _arm_action_cfg("right_robot")
    right_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action_cfg("right_robot")


@configclass
class BimanualYamObservationsCfg:
    """Joint state and previous-action policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated bimanual proprioceptive observations."""

        left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_robot")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("left_robot")})
        right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_robot")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("right_robot")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


_CAMERA_WIDTH = 1280
_CAMERA_HEIGHT = 720
_WRIST_CAMERA_POSITION = (-0.0107, 0.079729, 0.066021)
_WRIST_CAMERA_ROTATION_XYZW = (0.423, 0.0, 0.0, 0.906)
_TOP_CAMERA_OFFSET_FROM_MIDPOINT = (0.335, 0.0, 0.93732053)
_TOP_CAMERA_ROTATION_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
_LINK_SIX_SUFFIX = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"


def _d405_camera(
    prim_path: str,
    position: tuple[float, float, float] = _WRIST_CAMERA_POSITION,
    rotation_xyzw: tuple[float, float, float, float] = _WRIST_CAMERA_ROTATION_XYZW,
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
    """One overhead camera and one wrist camera on each YAM."""

    left_wrist_camera: CameraCfg = _d405_camera(f"{{ENV_REGEX_NS}}/YamLeft{_LINK_SIX_SUFFIX}/left_wrist_camera")
    right_wrist_camera: CameraCfg = _d405_camera(f"{{ENV_REGEX_NS}}/YamRight{_LINK_SIX_SUFFIX}/right_wrist_camera")
    top_camera: CameraCfg = _d405_camera(
        "{ENV_REGEX_NS}/top_camera",
        position=_TOP_CAMERA_OFFSET_FROM_MIDPOINT,
        rotation_xyzw=_TOP_CAMERA_ROTATION_XYZW,
    )

    def set_robot_prim_paths(self, left: str, right: str) -> None:
        """Attach the wrist cameras to articulation roots in the current scene."""
        self.left_wrist_camera.prim_path = f"{left}{_LINK_SIX_SUFFIX}/left_wrist_camera"
        self.right_wrist_camera.prim_path = f"{right}{_LINK_SIX_SUFFIX}/right_wrist_camera"

    def set_robot_mount_positions(
        self,
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> None:
        """Place the overhead camera relative to the robot pair midpoint."""
        midpoint = tuple((float(left_axis) + float(right_axis)) * 0.5 for left_axis, right_axis in zip(left, right))
        self.top_camera.offset.pos = tuple(
            midpoint_axis + offset_axis
            for midpoint_axis, offset_axis in zip(midpoint, _TOP_CAMERA_OFFSET_FROM_MIDPOINT)
        )


@register_asset
class BimanualYamEmbodiment(EmbodimentBase):
    """Two fixed-base YAM manipulators controlled by absolute joint targets."""

    name = "yam_bimanual"
    tags = ["embodiment", "yam", "bimanual"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        left_position: tuple[float, float, float] = YAM_DEFAULT_LEFT_POSITION,
        right_position: tuple[float, float, float] = YAM_DEFAULT_RIGHT_POSITION,
        left_yaw: float = 0.0,
        right_yaw: float = 0.0,
    ) -> None:
        """Initialize the fixed-layout bimanual YAM embodiment."""
        super().__init__(
            enable_cameras=enable_cameras,
            concatenate_observation_terms=True,
            arm_mode=ArmMode.DUAL_ARM,
        )
        self.scene_config = BimanualYamSceneCfg(
            left_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/YamLeft", left_position, left_yaw),
            right_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/YamRight", right_position, right_yaw),
        )
        self.action_config = BimanualYamActionsCfg()
        self.observation_config = BimanualYamObservationsCfg()
        self.camera_config = BimanualYamCameraCfg() if enable_cameras else None
        if self.camera_config is not None:
            self.camera_config.use_tiled_camera = False
            self.camera_config.set_robot_prim_paths("{ENV_REGEX_NS}/YamLeft", "{ENV_REGEX_NS}/YamRight")
            self.camera_config.set_robot_mount_positions(left_position, right_position)
            self.add_camera_variations(self.camera_config)

    def get_scene_key(self) -> str:
        """Return the left articulation as the primary scene key."""
        return "left_robot"

    def get_initial_pose(self) -> Pose:
        """Return the midpoint pose of the fixed bimanual base layout."""
        left = self.scene_config.left_robot.init_state.pos
        right = self.scene_config.right_robot.init_state.pos
        midpoint = tuple((float(left_axis) + float(right_axis)) * 0.5 for left_axis, right_axis in zip(left, right))
        return Pose(position_xyz=midpoint)

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update both YAM articulations' initial joint positions."""
        self.scene_config.left_robot.init_state.joint_pos.update(joint_pos)
        self.scene_config.right_robot.init_state.joint_pos.update(joint_pos)

    def get_command_body_name(self) -> str:
        return "link_6"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "link_6"
