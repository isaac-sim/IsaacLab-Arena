# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Expand independent Ability Hand joint angles to the full 10-DOF URDF set.

Dex-retargeting optimizes the six independent joints (four finger q1 + both thumb
joints). The four finger q2 joints are 4-bar linkage mimics of q1 and are derived
here before the teleop action tensor is assembled.
"""

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIO, RetargeterIOType
from isaacteleop.retargeting_engine.tensor_types import FloatType
from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType

# From ihmc_hands_ros2 AbilityHandModel / URDF mimic tags.
_ABILITY_HAND_Q2_MULTIPLIER = 1.05851325
_ABILITY_HAND_Q2_OFFSET = 0.72349796


class AbilityHandJointExpandRetargeter(BaseRetargeter):
    """Map six independent Ability Hand joints to all ten URDF finger joints."""

    def __init__(
        self,
        independent_joint_names: list[str],
        full_joint_names: list[str],
        name: str,
    ) -> None:
        self._independent_joint_names = independent_joint_names
        self._full_joint_names = full_joint_names
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "independent_hand_joints",
                [FloatType(joint_name) for joint_name in self._independent_joint_names],
            )
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "full_hand_joints",
                [FloatType(joint_name) for joint_name in self._full_joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        independent = inputs["hand_joints"]
        expanded = outputs["hand_joints"]
        values = {name: float(independent[i]) for i, name in enumerate(self._independent_joint_names)}

        for i, joint_name in enumerate(self._full_joint_names):
            if joint_name.endswith("_q2") and "thumb" not in joint_name:
                q1_name = joint_name.replace("_q2", "_q1")
                q1 = values.get(q1_name, 0.0)
                expanded[i] = _ABILITY_HAND_Q2_MULTIPLIER * q1 + _ABILITY_HAND_Q2_OFFSET
            else:
                expanded[i] = values.get(joint_name, 0.0)
