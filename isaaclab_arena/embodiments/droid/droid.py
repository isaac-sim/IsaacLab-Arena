# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import functools
import torch
import warnings
from abc import ABC
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
from isaaclab_arena.embodiments.droid.observations import arm_joint_pos, ee_pos, ee_quat, gripper_pos
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import franka_stack_events
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation

# The base stand's x/y footprint.
_STAND_FOOTPRINT_SCALE_XY: tuple[float, float] = (1.2, 1.2)
# The default stand height.
_DEFAULT_STAND_HEIGHT_M: float = 1.35
_FALLBACK_STAND_UNIT_HEIGHT_M: float = 0.795


class DroidEmbodimentBase(EmbodimentBase, ABC):
    """Abstract base class for DROID embodiments (https://droid-dataset.github.io/droid/docs/hardware-setup).

    Includes Franka with robotiq gripper and specific set of cameras.
    Subclasses must set ``self.action_config`` to a concrete action configuration.
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
        stand_height_m: float = _DEFAULT_STAND_HEIGHT_M,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = DroidSceneCfg()
        # ``stand_height_m`` is an absolute height in meters; convert it to the z-scale the USD needs.
        stand_unit_height = _stand_unit_height_m(self.scene_config.stand.spawn.usd_path)
        self.scene_config.stand.spawn.scale = (*_STAND_FOOTPRINT_SCALE_XY, stand_height_m / stand_unit_height)
        # Lift the robot base (and stand) so a taller/shorter stand keeps its bottom on the floor.
        self._robot_base_z_offset = stand_height_m - _DEFAULT_STAND_HEIGHT_M
        self.scene_config.robot.init_state.pos = self._lift_z(self.scene_config.robot.init_state.pos)
        self.scene_config.stand.init_state.pos = self._lift_z(self.scene_config.stand.init_state.pos)
        self.action_config = None
        self.camera_config = DroidCameraCfg()
        self.observation_config = DroidObservationsCfg()
        self.event_config = DroidEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = None
        self.mimic_env = None
        self.add_variation(CameraExtrinsicsVariation(camera_name="wrist_camera"))

    def _lift_z(self, pos: tuple[float, float, float]) -> tuple[float, float, float]:
        """Return ``pos`` shifted up by the stand-height-driven robot base offset."""
        return (pos[0], pos[1], pos[2] + self._robot_base_z_offset)

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        # We override the default initial pose setting function in order to also set the initial pose
        # of the stand, and to re-apply the stand-height lift on top of the requested pose (the base
        # implementation overwrites init_state.pos with the raw pose).
        scene_config = super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)
        if scene_config is None or not hasattr(scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        scene_config.robot.init_state.pos = self._lift_z(pose.position_xyz)
        scene_config.stand.init_state.pos = self._lift_z(pose.position_xyz)
        scene_config.stand.init_state.rot = pose.rotation_xyzw

        return scene_config

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
        stand_height_m: float = _DEFAULT_STAND_HEIGHT_M,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
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
        stand_height_m: float = _DEFAULT_STAND_HEIGHT_M,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
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
        stand_height_m: float = _DEFAULT_STAND_HEIGHT_M,
    ):
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            stand_height_m,
        )
        self.action_config = DroidAbsoluteJointPositionActionsCfg()


@configclass
class DroidSceneCfg:
    """Additions to the scene configuration coming from the Franka embodiment."""

    # The robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
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
    # The stand for the franka
    # TODO(alexmillane, 2025-07-28): We probably want to make the stand an optional addition.
    stand: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.05, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/robots/franka_stand_grey.usda"
            ),
            scale=(*_STAND_FOOTPRINT_SCALE_XY, _DEFAULT_STAND_HEIGHT_M / _FALLBACK_STAND_UNIT_HEIGHT_M),
            activate_contact_sensors=False,
        ),
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
        prim_path="{ENV_REGEX_NS}/Robot/external_camera",
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
        prim_path="{ENV_REGEX_NS}/Robot/external_camera_2",
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


@functools.cache
def _stand_unit_height_m(usd_path: str) -> float:
    """Native (scale=1.0) z-height of the stand USD in meters, cached per asset path.

    Falls back to ``_FALLBACK_STAND_UNIT_HEIGHT_M`` if the asset cannot be opened or measured.
    """
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(usd_path)
        assert stage is not None, f"could not open stand USD: {usd_path}"
        root_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root_prim)
        height = bound.ComputeAlignedRange().GetSize()[2]
        assert height > 0.0, f"non-positive stand height {height} from {usd_path}"
        return height
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the measured constant
        warnings.warn(
            f"Falling back to {_FALLBACK_STAND_UNIT_HEIGHT_M} m for the stand height; "
            f"could not measure {usd_path}: {exc!r}"
        )
        return _FALLBACK_STAND_UNIT_HEIGHT_M
