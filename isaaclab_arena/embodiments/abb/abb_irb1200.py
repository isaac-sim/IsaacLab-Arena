# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

_ABB_IRB1200_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)

_ABB_IRB1200_READY_POSE = {
    "joint_1": 0.0,
    "joint_2": -0.35,
    "joint_3": 0.65,
    "joint_4": 0.0,
    "joint_5": 0.75,
    "joint_6": 0.0,
}


def _find_irb1200_usd_path(require_exists: bool = False) -> str:
    relative_paths = (
        Path("isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.usda"),
        Path("references/abb/abb_irb1200_support/usd/irb1200_7_70/irb1200_7_70.usda"),
    )
    for parent in Path(__file__).resolve().parents:
        for relative_path in relative_paths:
            candidate = parent / relative_path
            if candidate.exists():
                return str(candidate)
    fallback_path = Path(__file__).resolve().parents[2] / relative_paths[0]
    if require_exists:
        search_locations = ", ".join(str(path) for path in relative_paths)
        raise FileNotFoundError(f"ABB IRB1200 USD not found. Searched: {search_locations}")
    return str(fallback_path)


def find_irb1200_robotiq_2f140_usd_path(require_exists: bool = False) -> str:
    relative_paths = (
        Path("isaaclab_arena/assets/robots/abb/irb1200_7_70_robotiq_2f140/irb1200_7_70.usda"),
    )
    for parent in Path(__file__).resolve().parents:
        for relative_path in relative_paths:
            candidate = parent / relative_path
            if candidate.exists():
                return str(candidate)
    fallback_path = Path(__file__).resolve().parents[2] / relative_paths[0]
    if require_exists:
        search_locations = ", ".join(str(path) for path in relative_paths)
        raise FileNotFoundError(f"ABB IRB1200 + Robotiq 2F-140 USD not found. Searched: {search_locations}")
    return str(fallback_path)


def find_irb1200_robotiq_2f140_physx_usd_path(require_exists: bool = False) -> str:
    relative_paths = (
        Path("isaaclab_arena/assets/robots/abb/irb1200_7_70_robotiq_2f140_physx/irb1200_7_70.usda"),
    )
    for parent in Path(__file__).resolve().parents:
        for relative_path in relative_paths:
            candidate = parent / relative_path
            if candidate.exists():
                return str(candidate)
    fallback_path = Path(__file__).resolve().parents[2] / relative_paths[0]
    if require_exists:
        search_locations = ", ".join(str(path) for path in relative_paths)
        raise FileNotFoundError(f"ABB IRB1200 + Robotiq 2F-140 PhysX USD not found. Searched: {search_locations}")
    return str(fallback_path)


@register_asset
class ABBIRB1200IKEmbodiment(EmbodimentBase):
    """ABB IRB1200 embodiment using relative differential IK."""

    name = "abb_irb1200_ik"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = ABBIRB1200SceneCfg()
        usd_path = _find_irb1200_usd_path(require_exists=True)
        self.scene_config.robot.spawn = self.scene_config.robot.spawn.replace(usd_path=usd_path)
        print(f"[INFO] ABB IRB1200 USD: {usd_path}")
        self.observation_config = ABBIRB1200ObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.reward_config = ABBIRB1200RewardsCfg()
        self.action_config = ABBIRB1200IKActionCfg()

        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        expected_joint_count = len(_ABB_IRB1200_JOINT_NAMES)
        assert (
            len(initial_joint_pose) == expected_joint_count
        ), f"expected {expected_joint_count} joint positions, got {len(initial_joint_pose)}"
        self.scene_config.robot.init_state = self.scene_config.robot.init_state.replace(
            joint_pos=dict(zip(_ABB_IRB1200_JOINT_NAMES, initial_joint_pose))
        )

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_teleop_target_frame_prim_path(self) -> str:
        return "/World/envs/env_0/Robot/Geometry/base_link"


@configclass
class ABBIRB1200SceneCfg:
    """Scene additions for ABB IRB1200."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_find_irb1200_usd_path(),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=_ABB_IRB1200_READY_POSE,
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=150.0,
                velocity_limit=4.0,
                stiffness=500.0,
                damping=50.0,
            ),
        },
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6",
                name="end_effector",
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        marker_cfg.prim_path = "/Visuals/ABBIRB1200FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class ABBIRB1200IKActionCfg:
    """6D relative pose action for ABB IRB1200."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        body_name="link_6",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.4,
    )


@configclass
class ABBIRB1200ObservationsCfg:
    """Observation specifications for the ABB IRB1200 embodiment."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ABBIRB1200RewardsCfg:
    """Small regularization rewards so ManagerBased env construction has reward terms."""

    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})
