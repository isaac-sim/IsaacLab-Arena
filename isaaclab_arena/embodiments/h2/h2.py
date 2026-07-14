# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree H2 embodiment bringup.

This module is intentionally small for the first H2 integration slices: it
registers a spawn-only ``h2_debug`` embodiment with H2 joint metadata so we can
validate that the H2 USD loads in Arena before adding IK, WBC, teleop, or policy
code.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.no_embodiment import EmptyActionsCfg
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

ROBOT_MENAGERIE_ROOT_ENV_VAR = "ROBOT_MENAGERIE_ROOT"
"""Optional root of a robot_menagerie checkout/cache containing ``unitree/h2``."""

H2_ROOT_FRAME_NAME = "pelvis"
"""H2 base/root link name in the robot_menagerie asset."""

H2_LEFT_WRIST_LINK_NAME = "left_wrist_yaw_link"
"""Terminal left wrist link used as the left hand frame for H2 bringup."""

H2_RIGHT_WRIST_LINK_NAME = "right_wrist_yaw_link"
"""Terminal right wrist link used as the right hand frame for H2 bringup."""

H2_HAND_FRAME_NAMES = {
    "left": H2_LEFT_WRIST_LINK_NAME,
    "right": H2_RIGHT_WRIST_LINK_NAME,
}
"""H2 hand-frame names keyed by arm side."""

H2_LEFT_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
)
"""H2 left leg joint names."""

H2_RIGHT_LEG_JOINT_NAMES = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
)
"""H2 right leg joint names."""

H2_LEG_JOINT_NAMES = H2_LEFT_LEG_JOINT_NAMES + H2_RIGHT_LEG_JOINT_NAMES
"""H2 leg joint names from robot_menagerie/unitree/h2/urdf/H2_simple_colliders.urdf."""

H2_WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
"""H2 waist joint names."""

H2_HEAD_JOINT_NAMES = (
    "head_pitch_joint",
    "head_yaw_joint",
)
"""H2 head joint names."""

H2_LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
"""H2 left arm joint names."""

H2_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""H2 right arm joint names."""

H2_ARM_JOINT_NAMES = H2_LEFT_ARM_JOINT_NAMES + H2_RIGHT_ARM_JOINT_NAMES
"""H2 arm joint names."""

H2_JOINT_NAMES = H2_LEG_JOINT_NAMES + H2_WAIST_JOINT_NAMES + H2_HEAD_JOINT_NAMES + H2_ARM_JOINT_NAMES
"""All actuated H2 joint names expected by the debug embodiment."""

H2_JOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "left_leg": H2_LEFT_LEG_JOINT_NAMES,
    "right_leg": H2_RIGHT_LEG_JOINT_NAMES,
    "legs": H2_LEG_JOINT_NAMES,
    "waist": H2_WAIST_JOINT_NAMES,
    "head": H2_HEAD_JOINT_NAMES,
    "left_arm": H2_LEFT_ARM_JOINT_NAMES,
    "right_arm": H2_RIGHT_ARM_JOINT_NAMES,
    "arms": H2_ARM_JOINT_NAMES,
    "upper_body": H2_WAIST_JOINT_NAMES + H2_HEAD_JOINT_NAMES + H2_ARM_JOINT_NAMES,
    "body": H2_JOINT_NAMES,
}
"""Named H2 joint groups used by debug bringup and later controller wiring."""

H2_DEFAULT_JOINT_POS = {joint_name: 0.0 for joint_name in H2_JOINT_NAMES}
"""Neutral H2 joint pose used for spawn/debug rollouts."""


# TODO(pulkitg): Move the H2 USD asset to Omniverse Nucleus once the asset is
# published, then replace the robot_menagerie path resolver with the Nucleus URL.
def _robot_menagerie_root() -> Path:
    """Return the configured robot_menagerie root inside the container."""

    return Path(os.environ.get(ROBOT_MENAGERIE_ROOT_ENV_VAR, "/workspaces/robot_menagerie"))


def _candidate_h2_usd_paths() -> tuple[Path, ...]:
    """Return supported H2 USD locations, ordered by preference."""

    candidates: list[Path] = []
    relative_exact_paths = (
        "unitree/h2/usd/H2_simple_colliders.usd",
        "unitree/h2/usd/H2.usd",
    )

    root = _robot_menagerie_root()
    candidates.extend(root / relative_path for relative_path in relative_exact_paths)
    return tuple(dict.fromkeys(candidates))


def resolve_h2_usd_path() -> str:
    """Resolve the H2 debug USD path.

    Use ``ROBOT_MENAGERIE_ROOT`` if set, falling back to
    ``/workspaces/robot_menagerie``. If no USD exists there, return the expected
    simple-collider USD path so any later spawn failure names the missing mount
    location.
    """

    candidates = _candidate_h2_usd_paths()
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def h2_debug_usd_exists() -> bool:
    """Return whether the currently resolved H2 debug USD exists locally."""

    return Path(resolve_h2_usd_path()).is_file()


H2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=resolve_h2_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=H2_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
            effort_limit=360.0,
            velocity_limit=20.0,
            stiffness=100.0,
            damping=1.0,
            armature=0.03,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit={
                ".*_ankle_roll_joint": 19.0,
                ".*_ankle_pitch_joint": 66.88,
            },
            velocity_limit={
                ".*_ankle_roll_joint": 100.7,
                ".*_ankle_pitch_joint": 28.61,
            },
            stiffness=100.0,
            damping=1.0,
            armature=0.03,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit={
                "waist_yaw_joint": 120.0,
                "waist_roll_joint": 180.0,
                "waist_pitch_joint": 180.0,
            },
            velocity_limit=28.375,
            stiffness=100.0,
            damping=1.0,
            armature=0.03,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=100.0,
            damping=1.0,
            armature=0.03,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 130.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 60.0,
                ".*_elbow_joint": 60.0,
                ".*_wrist_roll_joint": 60.0,
                ".*_wrist_pitch_joint": 10.0,
                ".*_wrist_yaw_joint": 10.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 21.9,
                ".*_shoulder_roll_joint": 18.7,
                ".*_shoulder_yaw_joint": 18.7,
                ".*_elbow_joint": 18.7,
                ".*_wrist_roll_joint": 18.7,
                ".*_wrist_pitch_joint": 37.7,
                ".*_wrist_yaw_joint": 37.7,
            },
            stiffness=100.0,
            damping=1.0,
            armature=0.03,
        ),
    },
)
"""Free-root H2 articulation config used as the base for H2 embodiments."""

H2_DEBUG_CFG = H2_CFG.copy()
# ``h2_debug`` has no balance controller yet; pin the root so asset inspection
# in the UI does not free-fall/collapse into the ground.
H2_DEBUG_CFG.spawn.articulation_props.fix_root_link = True


@register_asset
class H2DebugEmbodiment(EmbodimentBase):
    """Spawn-only Unitree H2 embodiment for asset bringup."""

    name = "h2_debug"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        **kwargs,
    ):
        # ``kwargs`` keeps this debug embodiment usable in generic envs that pass
        # camera/control options meant for richer humanoid embodiments.
        del kwargs
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = H2DebugSceneCfg()
        self.action_config = EmptyActionsCfg()
        self.observation_config = H2DebugObservationsCfg()
        self.event_config = H2DebugEventCfg()

    def get_teleop_target_frame_prim_path(self) -> str | None:
        return f"/World/envs/env_0/Robot/{H2_ROOT_FRAME_NAME}"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.LEFT:
            return H2_LEFT_WRIST_LINK_NAME
        if arm_mode in (ArmMode.RIGHT, ArmMode.SINGLE_ARM):
            return H2_RIGHT_WRIST_LINK_NAME
        return ""


@register_asset
class H2DebugJointPositionEmbodiment(H2DebugEmbodiment):
    """Fixed-root H2 debug embodiment with direct joint-position actions."""

    name = "h2_debug_joint_pos"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        **kwargs,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode, **kwargs)
        self.action_config = H2DebugJointPositionActionsCfg()


@configclass
class H2DebugSceneCfg:
    """Scene additions for the H2 debug embodiment."""

    robot: ArticulationCfg = H2_DEBUG_CFG.copy()


@configclass
class H2DebugJointPositionActionsCfg:
    """Direct joint-position action for fixed-root H2 debug rollouts."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(H2_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class H2DebugObservationsCfg:
    """Minimal observations for H2 asset/debug rollouts."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class H2DebugEventCfg:
    """Reset events for H2 debug rollouts."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")
