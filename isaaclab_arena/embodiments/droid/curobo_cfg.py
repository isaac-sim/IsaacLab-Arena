# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo description for the DROID robot (Franka + Robotiq 2F-85)."""

import torch

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.embodiments.common.curobo_cfg import CuroboEmbodimentCfg

# cuRobo reads real on-disk files, so the config/URDF are Nucleus/S3 paths the utils download at
# planner-construction time.
_DROID_CUROBO_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/robot_library/droid/droid_fixed_mimic_joint"

DROID_CUROBO_CFG = CuroboEmbodimentCfg(
    robot_cfg_template=f"{_DROID_CUROBO_ASSET_DIR}/franka_robotiq_2f_85_zero_curobo.yml",
    robot_urdf=f"{_DROID_CUROBO_ASSET_DIR}/urdf/franka_robotiq_2f_85_zero.urdf",
    robot_name="franka_robotiq",
    ee_link_name="base_link",
    gripper_joint_names=["finger_joint"],
    gripper_open_joint_pos={"finger_joint": 0.0},
    gripper_closed_joint_pos={"finger_joint": float(torch.pi / 4)},
    hand_link_names=[
        "base_link",
        "left_inner_finger",
        "left_inner_knuckle",
        "left_outer_finger",
        "left_outer_knuckle",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_finger",
        "right_outer_knuckle",
    ],
)
