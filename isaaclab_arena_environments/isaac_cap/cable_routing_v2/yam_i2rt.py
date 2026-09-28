# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cable Easy's upstream I2RT YAM embodiment."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import ActionTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from .embodiment.actions import ContinuousJointPositionZeroToOneActionCfg, FiniteJointPositionActionCfg
from .embodiment.cameras import BimanualYamCameraCfg as IndustrialBimanualYamCameraCfg

_ASSET_ROOT = (
    f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/latest/cable_routing/assets/yam_i2rt"
)
ASSET_PATH = f"{_ASSET_ROOT}/yam_i2rt.usda"
ARM_HOME = (0.0, 1.4, 1.2, 0.0, 1.1, 0.0)
GRIPPER_OPEN_POSITION = 0.0475
GRIPPER_CLOSED_POSITION = 0.0
GRASP_SITE_OFFSET = (0.0, 0.0, -0.1347)
GRASP_SITE_ROTATION_XYZW = (1.0, 0.0, 0.0, 0.0)
BASE_HALF_SEPARATION = 0.31
_WRIST_CAMERA_POSITION = (-0.06999901687071203, -0.0003709947894147738, 0.03)
_WRIST_CAMERA_ROTATION_XYZW = (
    -0.002619818894631038,
    0.9886178461056714,
    0.00039862390329551946,
    0.15042517079706494,
)
_WRIST_CAMERA_WIDTH = 640
_WRIST_CAMERA_HEIGHT = 480
_WRIST_CAMERA_VERTICAL_FOV_DEG = 58.0


def arm_joint_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_joint{index}" for index in range(1, 7))


def gripper_joint_names(side: str) -> tuple[str, str]:
    return (f"{side}_joint7", f"{side}_joint8")


def gripper_body_name(side: str) -> str:
    return f"{side}_gripper"


def _link_six_path(side: str) -> str:
    links = "/".join(f"{side}_link{index}" for index in range(1, 7))
    return f"{{ENV_REGEX_NS}}/Yam/Geometry/{side}_base/{links}/{side}_gripper"


def _actuators(side: str) -> dict[str, ImplicitActuatorCfg]:
    return {
        "arm_joints_1_3": ImplicitActuatorCfg(
            joint_names_expr=[f"{side}_joint[1-3]"],
            stiffness=300.0,
            damping=30.0,
            effort_limit_sim=27.0,
        ),
        "arm_joints_4_6": ImplicitActuatorCfg(
            joint_names_expr=[f"{side}_joint[4-6]"],
            stiffness=300.0,
            damping=30.0,
            effort_limit_sim=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(gripper_joint_names(side)),
            stiffness=600.0,
            damping=20.0,
            effort_limit_sim=80.0,
        ),
    }


def _articulation(side: str, y: float) -> ArticulationCfg:
    joint_pos = dict(zip(arm_joint_names(side), ARM_HOME, strict=True))
    joint_pos.update({name: GRIPPER_OPEN_POSITION for name in gripper_joint_names(side)})
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Yam",
        articulation_root_prim_path=f"/Geometry/{side}_base",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, y, 0.0),
            joint_pos=joint_pos,
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=_actuators(side),
    )


@configclass
class CableRoutingYamI2rtSceneCfg:
    """Spawn the bimanual model once and expose one articulation view per arm."""

    yam_model: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Yam",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_PATH,
            activate_contact_sensors=True,
            copy_from_source=False,
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
        ),
    )
    left_robot: ArticulationCfg = _articulation("left", BASE_HALF_SEPARATION)
    right_robot: ArticulationCfg = _articulation("right", -BASE_HALF_SEPARATION)


def _arm_action(side: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=f"{side}_robot",
        joint_names=list(arm_joint_names(side)),
        preserve_order=True,
        use_default_offset=False,
    )


def _gripper_action(side: str) -> ContinuousJointPositionZeroToOneActionCfg:
    names = gripper_joint_names(side)
    return ContinuousJointPositionZeroToOneActionCfg(
        asset_name=f"{side}_robot",
        joint_names=list(names),
        open_command_expr={name: GRIPPER_OPEN_POSITION for name in names},
        close_command_expr={name: GRIPPER_CLOSED_POSITION for name in names},
    )


@configclass
class CableRoutingYamI2rtActionsCfg:
    """Keep Arena's left-block/right-block 14-dimensional action contract."""

    left_arm_action: ActionTermCfg = _arm_action("left")
    left_gripper_action: ActionTermCfg = _gripper_action("left")
    right_arm_action: ActionTermCfg = _arm_action("right")
    right_gripper_action: ActionTermCfg = _gripper_action("right")


def _indices(names: list[str], expected: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in expected]
    except ValueError as error:
        raise ValueError(f"YAM articulation is missing one of {expected!r}") from error


def _arm_joint_pos(env, asset_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    indices = _indices(robot.data.joint_names, arm_joint_names(side))
    return wp.to_torch(robot.data.joint_pos)[:, indices]


def _gripper_pos(env, asset_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    indices = _indices(robot.data.joint_names, gripper_joint_names(side))
    position = wp.to_torch(robot.data.joint_pos)[:, indices].mean(dim=1, keepdim=True)
    closedness = (GRIPPER_OPEN_POSITION - position) / GRIPPER_OPEN_POSITION
    return torch.clamp(closedness, min=0.0, max=1.0)


def _ee_pos(env, asset_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    index = _indices(robot.data.body_names, (gripper_body_name(side),))[0]
    body_pos = wp.to_torch(robot.data.body_link_pos_w)[:, index, :]
    body_quat = wp.to_torch(robot.data.body_link_quat_w)[:, index, :]
    offset = body_pos.new_tensor(GRASP_SITE_OFFSET).expand(body_pos.shape[0], -1)
    return body_pos + math_utils.quat_apply(body_quat, offset)


def _ee_quat(env, asset_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    index = _indices(robot.data.body_names, (gripper_body_name(side),))[0]
    body_quat = wp.to_torch(robot.data.body_link_quat_w)[:, index, :]
    offset = body_quat.new_tensor(GRASP_SITE_ROTATION_XYZW).expand(body_quat.shape[0], -1)
    return math_utils.quat_mul(body_quat, offset)


@configclass
class CableRoutingYamI2rtObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            for side in ("left", "right"):
                asset_cfg = SceneEntityCfg(f"{side}_robot")
                setattr(
                    self,
                    f"{side}_joint_pos",
                    ObsTerm(
                        func=_arm_joint_pos,
                        params={"asset_cfg": asset_cfg, "side": side},
                    ),
                )
                setattr(
                    self,
                    f"{side}_gripper_pos",
                    ObsTerm(
                        func=_gripper_pos,
                        params={"asset_cfg": asset_cfg, "side": side},
                    ),
                )
                setattr(
                    self,
                    f"{side}_eef_pos",
                    ObsTerm(
                        func=_ee_pos,
                        params={"asset_cfg": asset_cfg, "side": side},
                    ),
                )
                setattr(
                    self,
                    f"{side}_eef_quat",
                    ObsTerm(
                        func=_ee_quat,
                        params={"asset_cfg": asset_cfg, "side": side},
                    ),
                )
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CableRoutingYamI2rtEventCfg:
    reset_left_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("left_robot"),
        },
    )
    reset_right_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("right_robot"),
        },
    )


class CableRoutingYamI2rtEmbodiment(EmbodimentBase):
    """The exact upstream YAM I2RT model, isolated to Cable Easy."""

    name = "cable_routing_yam_i2rt"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        *,
        model_position: tuple[float, float, float],
        enable_cameras: bool = False,
        use_tiled_cameras: bool = False,
        cable_camera_width: int = 1280,
        medium: bool = False,
    ) -> None:
        if type(cable_camera_width) is not int or not 1280 <= cable_camera_width <= 4096:
            raise ValueError("cable camera width must be an integer from 1280 to 4096")
        super().__init__(
            enable_cameras=enable_cameras,
            concatenate_observation_terms=True,
            arm_mode=ArmMode.DUAL_ARM,
        )
        self.scene_config = CableRoutingYamI2rtSceneCfg()
        if medium:
            self.scene_config.yam_model.spawn.usd_path = f"{_ASSET_ROOT}/yam_i2rt_medium.usda"
        self.scene_config.yam_model.init_state.pos = model_position
        for side, y_offset in (
            ("left", BASE_HALF_SEPARATION),
            ("right", -BASE_HALF_SEPARATION),
        ):
            robot = getattr(self.scene_config, f"{side}_robot")
            robot.init_state.pos = (
                model_position[0],
                model_position[1] + y_offset,
                model_position[2],
            )

        self.action_config = CableRoutingYamI2rtActionsCfg()
        self.observation_config = CableRoutingYamI2rtObservationsCfg()
        self.event_config = CableRoutingYamI2rtEventCfg()
        self.camera_config = IndustrialBimanualYamCameraCfg() if enable_cameras else None
        if self.camera_config is not None:
            self.camera_config.use_tiled_camera = use_tiled_cameras
            self.camera_config.use_cable_routing_rig()
            camera = self.camera_config.cable_camera
            # Preserve the native focal pixel scale when expanding coverage.
            camera.spawn.horizontal_aperture *= cable_camera_width / camera.width
            camera.width = cable_camera_width
            self.camera_config.top_camera.offset.rot = (
                math.sqrt(0.5),
                -math.sqrt(0.5),
                0.0,
                0.0,
            )
            self.camera_config.left_wrist_camera.prim_path = f"{_link_six_path('left')}/left_wrist_camera"
            self.camera_config.right_wrist_camera.prim_path = f"{_link_six_path('right')}/right_wrist_camera"
            for camera in (
                self.camera_config.left_wrist_camera,
                self.camera_config.right_wrist_camera,
            ):
                camera.offset.pos = _WRIST_CAMERA_POSITION
                camera.update_latest_camera_pose = True
                camera.offset.rot = _WRIST_CAMERA_ROTATION_XYZW
                camera.width = _WRIST_CAMERA_WIDTH
                camera.height = _WRIST_CAMERA_HEIGHT
                camera.spawn.focal_length = camera.spawn.vertical_aperture / (
                    2.0 * math.tan(math.radians(_WRIST_CAMERA_VERTICAL_FOV_DEG / 2.0))
                )
            # Match Berkeley RendererRTX.usd_film_back: USD pixel centers are
            # i+0.5, while the graph's centered OpenCV intrinsics use index i.
            # Omitting this shifts depth-edge samples relative to native.
            for camera in (
                self.camera_config.top_camera,
                self.camera_config.cable_camera,
                self.camera_config.left_wrist_camera,
                self.camera_config.right_wrist_camera,
            ):
                camera.spawn.horizontal_aperture_offset = -0.5 * camera.spawn.horizontal_aperture / camera.width
                camera.spawn.vertical_aperture_offset = 0.5 * camera.spawn.vertical_aperture / camera.height
            if medium:
                self.camera_config.top_camera = None
                self.camera_config.left_wrist_camera = None
                self.camera_config.right_wrist_camera = None

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        del arm_mode
        return gripper_body_name("right")

    def get_command_body_name(self) -> str:
        return gripper_body_name("right")
