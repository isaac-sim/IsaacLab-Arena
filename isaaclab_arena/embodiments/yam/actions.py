# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Actions for the shared bimanual YAM embodiment."""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, skew_symmetric_matrix

from .config import ARM_JOINT_NAMES, GRIPPER_CLOSED_POSITION, GRIPPER_JOINT_NAME, GRIPPER_OPEN_POSITION


class FiniteJointPositionAction(JointPositionAction):
    """Keep absolute joint targets finite and inside the soft limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize and constrain absolute joint-position commands."""
        finite_actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        super().process_actions(finite_actions)
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.where(torch.isfinite(self._processed_actions), self._processed_actions, default)
        self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


class NormalizedFiniteJointPositionAction(FiniteJointPositionAction):
    """Map a finite command in ``[0, 1]`` through the joint transform."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clamp the normalized command before applying its configured transform."""
        normalized_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        super().process_actions(normalized_actions)


@configclass
class FiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure finite absolute joint-position actions."""

    class_type: type[FiniteJointPositionAction] = FiniteJointPositionAction


@configclass
class NormalizedFiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure normalized finite joint-position actions."""

    class_type: type[NormalizedFiniteJointPositionAction] = NormalizedFiniteJointPositionAction


def _arm_action(asset_name: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


def _gripper_action(asset_name: str) -> NormalizedFiniteJointPositionActionCfg:
    return NormalizedFiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=[GRIPPER_JOINT_NAME],
        scale=GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION,
        offset=GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class BimanualYamActionsCfg:
    """Left arm/gripper followed by right arm/gripper in policy order."""

    left_arm_action: FiniteJointPositionActionCfg = _arm_action("left_robot")
    left_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("left_robot")
    right_arm_action: FiniteJointPositionActionCfg = _arm_action("right_robot")
    right_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("right_robot")


class YamTcpIKAction(DifferentialInverseKinematicsAction):
    """Use Isaac Lab IK with a rotated TCP lever arm and bounded joint targets."""

    def _compute_frame_jacobian(self):
        self._jacobian_b[:] = self.jacobian_b
        # Express the link_6-to-TCP lever arm in the base-frame Jacobian coordinates.
        _, q_B_TCP = self._compute_frame_pose()
        q_B_L = quat_mul(q_B_TCP, quat_inv(self._offset_rot))
        offset_b = quat_apply(q_B_L, self._offset_pos)
        self._jacobian_b[:, :3] += torch.bmm(-skew_symmetric_matrix(offset_b), self._jacobian_b[:, 3:])
        return self._jacobian_b

    def process_actions(self, actions):
        assert torch.isfinite(actions).all(), "YAM IK commands must be finite"
        command = actions.clone()
        if not self.cfg.controller.use_relative_mode:
            norm = command[:, 3:].norm(dim=-1, keepdim=True)
            assert (norm > 1e-6).all(), "YAM absolute pose commands need a nonzero xyzw quaternion"
            command[:, 3:] /= norm
        super().process_actions(command)

    def apply_actions(self):
        position, quaternion = self._compute_frame_pose()
        current = self._asset.data.joint_pos.torch[:, self._joint_ids]
        desired = self._ik_controller.compute(position, quaternion, self._compute_frame_jacobian(), current)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        desired = current + self.cfg.tracking_gain * (desired - current).clamp(-0.08, 0.08)
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.gravity_compensation_forces.torch
            # PhysX can expose a constrained root's six base entries in this dynamics buffer.
            base_columns = gravity.shape[-1] - self._asset.num_joints
            assert base_columns in (0, 6), "Unexpected YAM gravity-force layout"
            indices = [joint + base_columns for joint in self._joint_ids]
            stiffness = self._asset.data.joint_stiffness.torch[:, self._joint_ids]
            assert (stiffness > 0).all(), "YAM IK requires position-controlled arm actuators"
            # A position bias supplies gravity feedforward through the existing effort-limited drives.
            desired += gravity[:, indices] / stiffness
        desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
        self._asset.set_joint_position_target_index(target=desired, joint_ids=self._joint_ids)


@configclass
class YamTcpIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[YamTcpIKAction] = YamTcpIKAction
    tracking_gain: float = 1.0
    """Joint-space fraction of each bounded IK correction."""
    gravity_compensation: bool = True
    """Compensate model gravity using a position bias within the actuator's torque limit."""


def make_yam_ik_actions(relative=False):
    """Create left TCP/gripper then right TCP/gripper actions, each in its own arm base frame."""
    actions = BimanualYamActionsCfg()
    for side in ("left", "right"):
        setattr(
            actions,
            f"{side}_arm_action",
            YamTcpIKActionCfg(
                asset_name=f"{side}_robot",
                joint_names=ARM_JOINT_NAMES,
                body_name="link_6",
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, -0.044, 0.13)),
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=relative,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.02},
                ),
            ),
        )
    return actions
