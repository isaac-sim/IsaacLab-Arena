# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree H2 embodiment bringup.

This module is intentionally small for the first H2 integration slice: it only
registers a spawn-only ``h2_debug`` embodiment so we can validate that the H2 USD
loads in Arena before adding robot metadata, IK, WBC, teleop, or policy code.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
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


H2_DEBUG_CFG = ArticulationCfg(
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
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
        ),
    },
)


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
        return "/World/envs/env_0/Robot/pelvis"


@configclass
class H2DebugSceneCfg:
    """Scene additions for the H2 debug embodiment."""

    robot: ArticulationCfg = H2_DEBUG_CFG.copy()


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
