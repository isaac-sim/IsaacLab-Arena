# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Actions for the industrial FR3 Robotiq embodiments."""

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.droid.droid import BinaryJointPositionZeroToOneActionCfg

from .config import ARM_JOINT_NAMES, END_EFFECTOR_BODY_NAME, GRIPPER_CLOSED_ANGLE, GRIPPER_JOINT_NAME


class HoldingDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Preserve the last Cartesian target while the relative command is idle."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._hold_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        previous_pos = self._ik_controller.ee_pos_des.clone()
        previous_quat = self._ik_controller.ee_quat_des.clone()
        super().process_actions(actions)
        inactive = torch.linalg.vector_norm(actions, dim=1) <= self.cfg.hold_deadband
        restore = self._hold_initialized & inactive
        self._ik_controller.ee_pos_des[restore] = previous_pos[restore]
        self._ik_controller.ee_quat_des[restore] = previous_quat[restore]
        self._hold_initialized.fill_(True)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._hold_initialized[env_ids] = False


@configclass
class HoldingDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Relative IK that retains its Cartesian target while input is idle."""

    class_type: type[HoldingDifferentialInverseKinematicsAction] = HoldingDifferentialInverseKinematicsAction
    hold_deadband: float = 1.0e-6


@configclass
class IndustrialFr3RobotiqActionsCfg:
    """Seven ordered absolute arm targets and a zero-to-one gripper command."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        open_command_expr={GRIPPER_JOINT_NAME: 0.0},
        close_command_expr={GRIPPER_JOINT_NAME: GRIPPER_CLOSED_ANGLE},
    )


@configclass
class IndustrialFr3RobotiqDifferentialIKActionsCfg:
    """Relative Cartesian FR3 commands and a zero-to-one gripper command."""

    arm_action: ActionTermCfg = HoldingDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name=END_EFFECTOR_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        open_command_expr={GRIPPER_JOINT_NAME: 0.0},
        close_command_expr={GRIPPER_JOINT_NAME: GRIPPER_CLOSED_ANGLE},
    )
