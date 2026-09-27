# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot and scene configuration for the bimanual YAM embodiment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
END_EFFECTOR_BODY_NAME = "link_6"
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0

_DEFAULT_ARM_JOINT_POSITIONS = (0.0, 0.85, 0.60, 0.0, 0.0, 0.0)


def make_yam_articulation_cfg(
    prim_path: str,
    position: tuple[float, float, float],
    robot_usd_path: str,
) -> ArticulationCfg:
    """Build one fixed-base YAM articulation at the requested mount position."""
    joint_pos = dict(zip(ARM_JOINT_NAMES, _DEFAULT_ARM_JOINT_POSITIONS, strict=True))
    joint_pos[GRIPPER_JOINT_NAME] = GRIPPER_OPEN_POSITION
    joint_pos[PASSIVE_GRIPPER_JOINT_NAME] = -GRIPPER_OPEN_POSITION
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
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


@configclass
class BimanualYamSceneCfg:
    """Two independently addressable YAM articulations."""

    left_robot: ArticulationCfg | None = None
    right_robot: ArticulationCfg | None = None
