# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static scene and observation configurations for the selected FR3 asset."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootBaseCfg
from isaaclab.utils.configclass import configclass

ARM_JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
GRIPPER_JOINT_NAME = "left_driver_joint"
END_EFFECTOR_BODY_NAME = "robotiq_base"
# The authored driver upper limit is 51.5662 degrees.
GRIPPER_CLOSED_ANGLE = math.radians(51.5662)
_DROID_WORKING_HEIGHT_M = 1.35
_ROBOT_ON_CART_USD_PATH = (
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/newton_envs/cap_envs/gear_assembly/"
    "assets/industrial__fr3_robotiq_2f85_on_cart/industrial__fr3_robotiq_2f85_on_cart.usda"
)


@configclass
class IndustrialFr3RobotiqSceneCfg:
    """A fixed FR3; its computer-cart stand is composed into the robot USD."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROBOT_ON_CART_USD_PATH,
            activate_contact_sensors=True,
            articulation_props=ArticulationRootBaseCfg(fix_root_link=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, _DROID_WORKING_HEIGHT_M),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -math.pi / 5,
                "fr3_joint3": 0.0,
                "fr3_joint4": -4 * math.pi / 5,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3 * math.pi / 5,
                "fr3_joint7": 0.0,
                GRIPPER_JOINT_NAME: 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "fr3_joint_1_2": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-2]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=650.0,
                damping=100.0,
            ),
            "fr3_joint_3_4": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[3-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=650.0,
                damping=100.0,
            ),
            "fr3_joint_5_7": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=650.0,
                damping=100.0,
            ),
            "robotiq_driver": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_NAME],
                stiffness=100.0,
                effort_limit_sim=5.0,
                velocity_limit_sim=2.0,
                damping=10.0,
            ),
        },
    )


@configclass
class IndustrialFr3RobotiqEventCfg:
    """Restore the robot's configured working pose on every reset."""

    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class IndustrialFr3RobotiqObservationsCfg:
    """DROID-compatible keys for arm, gripper, and Robotiq base state."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            from .observations import arm_joint_pos, ee_pos, ee_quat, gripper_pos

            self.joint_pos = ObsTerm(func=arm_joint_pos)
            self.gripper_pos = ObsTerm(func=gripper_pos)
            self.eef_pos = ObsTerm(func=ee_pos)
            self.eef_quat = ObsTerm(func=ee_quat)
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
