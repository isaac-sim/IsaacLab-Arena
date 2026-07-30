# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from abc import ABC

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
from isaaclab_arena.embodiments.droid.observations import arm_joint_pos, ee_pos, ee_quat, gripper_pos
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import franka_stack_events
from isaaclab_arena.embodiments.robot_on_stand_utils import RobotPrimSpec, StandPrimSpec, compose_on_stand_usd
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

_DROID_ROBOT_PRIM = RobotPrimSpec(
    robot_usd_path=f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
    root_prim_path="/panda",
    robot_base_prim_name="panda_link0",
    stand_prim_name="stand_instanceable",
)
_DROID_STAND_PRIM = StandPrimSpec(
    stand_usd_path=f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/robots/franka_stand_grey.usda",
    ref_prim_path="/World/franka_table",
    payload_child_name="franka_table",
    footprint_translate_xyz=(-0.05, 0.0, 0.0),
    footprint_scale_xy=(1.2, 1.2),
    stand_default_height=1.35,
)


class DroidEmbodimentBase(EmbodimentBase, ABC):
    """Abstract base class for DROID embodiments (https://droid-dataset.github.io/droid/docs/hardware-setup).

    Includes Franka with robotiq gripper and specific set of cameras.
    Subclasses must set ``self.action_config`` to a concrete action configuration.

    ``initial_pose`` / ``set_initial_pose`` set the base of the robot in world frame.
    ``stand_height_m`` sets the height of the stand mesh under the robot base link,
    which changes how far the stand extends below the root link.
    When manually placing the robot on floor, ``set_initial_pose`` z value and
    ``stand_height_m`` should be adjusted together to keep the bottom of stand fixed.
    ``placement_bbox_stand_only`` uses the stand footprint for ``On`` / ``NextTo`` placement
    instead of the full robot+stand USD bounds.
    """

    name = "droid"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        stand_height_m: float = _DROID_STAND_PRIM.stand_default_height,
        placement_bbox_stand_only: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.stand_height_m = stand_height_m
        self.placement_bbox_stand_only = placement_bbox_stand_only
        self.scene_config = DroidSceneCfg()
        self.scene_config.robot.spawn.usd_path = compose_on_stand_usd(
            _DROID_ROBOT_PRIM,
            _DROID_STAND_PRIM,
            stand_height_m=stand_height_m,
            output_basename="droid_franka_robotiq_on_stand",
        )
        self.action_config = None
        self.camera_config = DroidCameraCfg()
        self.observation_config = DroidObservationsCfg()
        self.event_config = DroidEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = None
        self.mimic_env = None
        self.add_camera_variations(self.camera_config)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative placement bounds from the composed on-stand USD spawn.

        When ``placement_bbox_stand_only`` is True, bounds exclude the robot arm and cover
        the stand footprint only.
        """
        prim_path = _DROID_ROBOT_PRIM.stand_prim_path if self.placement_bbox_stand_only else None
        return super().get_bounding_box(prim_path=prim_path)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        self.event_config.init_franka_arm_pose.params["default_pose"] = initial_joint_pose

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


@register_asset
class DroidDifferentialIKEmbodiment(DroidEmbodimentBase):
    """Embodiment for the DROID setup with differential inverse kinematics action controller."""

    name = "droid_differential_ik"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        stand_height_m: float = _DROID_STAND_PRIM.stand_default_height,
        placement_bbox_stand_only: bool = False,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
            placement_bbox_stand_only,
        )
        self.action_config = DroidDifferentialIKActionsCfg()


@register_asset
class DroidRelativeJointPositionEmbodiment(DroidEmbodimentBase):
    """Embodiment for the DROID setup with relative joint position action controller."""

    name = "droid_rel_joint_pos"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        stand_height_m: float = _DROID_STAND_PRIM.stand_default_height,
        placement_bbox_stand_only: bool = False,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
            placement_bbox_stand_only,
        )
        self.action_config = DroidRelativeJointPositionActionsCfg()


@register_asset
class DroidAbsoluteJointPositionEmbodiment(DroidEmbodimentBase):
    """Embodiment for the DROID setup with absolute joint position actions."""

    name = "droid_abs_joint_pos"
    tags = ["embodiment", "default"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        stand_height_m: float = _DROID_STAND_PRIM.stand_default_height,
        placement_bbox_stand_only: bool = False,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
            placement_bbox_stand_only,
        )
        self.action_config = DroidAbsoluteJointPositionActionsCfg()


@configclass
class DroidSceneCfg:
    """Additions to the scene configuration coming from the Droid embodiment.

    The robot USD path is overwritten at embodiment construction via
    ``compose_on_stand_usd`` (cached local robot+stand assembly).
    """

    # The robot (stand is baked into the local on-stand USD, not a separate prim).
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DROID_ROBOT_PRIM.robot_usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 1),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * torch.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * torch.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * torch.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=None,
                damping=None,
                velocity_limit=1.0,
            ),
        },
    )

    # The end-effector frame marker
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


@configclass
class DroidDifferentialIKActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_link0",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": torch.pi / 4},
    )


@configclass
class DroidRelativeJointPositionActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        use_zero_offset=True,  # increment around current joint pos
        scale=0.5,  # scale factor for the action
    )
    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": torch.pi / 4},
    )


@configclass
class DroidAbsoluteJointPositionActionsCfg:
    """Absolute joint position actions."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": torch.pi / 4},
    )


@configclass
class DroidObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos)
        eef_pos = ObsTerm(func=ee_pos)
        eef_quat = ObsTerm(func=ee_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidEventCfg:
    """Configuration for Franka."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                0.0,  # panda_joint1
                -1 / 5 * torch.pi,  # panda_joint2
                0.0,  # panda_joint3
                -4 / 5 * torch.pi,  # panda_joint4
                0.0,  # panda_joint5
                3 / 5 * torch.pi,  # panda_joint6
                0.0,  # panda_joint7
                0.0,  # finger_joint
                0.0,  # right_outer_knuckle_joint
                0.0,  # right_inner_finger_joint
                0.0,  # right_inner_finger_knuckle_joint
                0.0,  # left_inner_finger_knuckle_joint
                0.0,  # left_inner_finger_joint
            ],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class DroidCameraCfg(ArenaCameraCfg):
    """Configuration for cameras. DROID cameras are mounted with pre-set poses."""

    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/external_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.57, 0.66), rot=(-0.195, 0.399, 0.805, -0.393), convention="opengl"),
    )
    external_camera_2: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/external_camera_2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, -0.57, 0.66), rot=(0.399, -0.195, -0.393, 0.805), convention="opengl"),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(0.570, 0.576, -0.409, -0.420), convention="opengl"
        ),
    )
