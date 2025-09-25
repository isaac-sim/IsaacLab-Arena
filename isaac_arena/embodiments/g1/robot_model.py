# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import yaml

from isaac_arena.embodiments.g1.g1_supplemental_info import G1SupplementalInfo


class RobotModel:
    def __init__(
        self,
        supplemental_info: G1SupplementalInfo | None = None,
    ):
        joints_order_path = os.path.join(
            os.path.dirname(__file__), "wbc_policy/config/loco_manip_g1_joints_order_43dof.yaml"
        )

        with open(joints_order_path) as f:
            self.wbc_g1_joints_order = yaml.safe_load(f)

        self.joint_to_dof_index = {}
        for name in self.wbc_g1_joints_order:
            self.joint_to_dof_index[name] = self.wbc_g1_joints_order[name]

        # Set up supplemental info if provided
        self.supplemental_info = supplemental_info
        print(f"self.supplemental_info: {self.supplemental_info}")
        self.num_dofs_body = len(self.supplemental_info.body_actuated_joints)
        self.num_dofs_hands = len(self.supplemental_info.left_hand_actuated_joints) + len(
            self.supplemental_info.right_hand_actuated_joints
        )
        if self.supplemental_info is not None:
            # Cache indices for joint groups, handling nested groups
            self._joint_group_indices = {}
            for group_name, group_info in self.supplemental_info.joint_groups.items():
                indices = []
                # Add indices for direct joints
                indices.extend([self.dof_index(name) for name in group_info["joints"]])
                # Add indices from subgroups
                for subgroup_name in group_info["groups"]:
                    indices.extend(self.get_joint_group_indices(subgroup_name))
                self._joint_group_indices[group_name] = sorted(set(indices))

        self.initial_body_pose = None

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom of the robot (floating base pose + joints)."""
        return self.num_dofs_body + self.num_dofs_hands

    @property
    def q_default(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        return self.initial_body_pose

    @property
    def joint_names(self) -> list[str]:
        """Get the names of the joints of the robot."""
        return list(self.joint_to_dof_index.keys())

    @property
    def num_joints(self) -> int:
        """Get the number of joints of the robot."""
        return len(self.joint_to_dof_index)

    def dof_index(self, joint_name: str) -> int:
        """
        Get the index in the degrees of freedom vector corresponding
        to the single-DoF joint with name `joint_name`.
        """
        if joint_name not in self.joint_to_dof_index:
            raise ValueError(
                f"Unknown joint name: '{joint_name}'. Available joints: {list(self.joint_to_dof_index.keys())}"
            )
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> list[int]:
        """
        Get the indices of body actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> list[int]:
        """
        Get the indices of hand actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.

        Args:
            side: String specifying which hand to get indices for ('left', 'right', or 'both')
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_joint_group_indices(self, group_names: str | set[str]) -> list[int]:
        """
        Get the indices of joints in one or more groups in the full configuration.
        Requires supplemental_info to be provided.
        The returned indices are sorted in ascending order, so that the joint ordering
        of the full model is preserved.

        Args:
            group_names: Either a single group name (str) or a set of group names (Set[str])

        Returns:
            List of joint indices in sorted order with no duplicates
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Convert single string to set for uniform handling
        if isinstance(group_names, str):
            group_names = {group_names}

        # Collect indices from all groups
        all_indices = set()
        for group_name in group_names:
            if group_name not in self._joint_group_indices:
                raise ValueError(f"Unknown joint group: {group_name}")
            all_indices.update(self._joint_group_indices[group_name])

        return sorted(all_indices)

    def get_body_actuated_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Get the configuration of body actuated joints from a full configuration.

        :param q: Configuration in full space
        :return: Configuration of body actuated joints
        """
        indices = self.get_body_actuated_joint_indices()

        return q[indices]

    def get_hand_actuated_joints(self, q: np.ndarray, side: str = "both") -> np.ndarray:
        """
        Get the configuration of hand actuated joints from a full configuration.

        Args:
            q: Configuration in full space
            side: String specifying which hand to get joints for ('left', 'right', or 'both')
        """
        indices = self.get_hand_actuated_joint_indices(side)
        return q[indices]
