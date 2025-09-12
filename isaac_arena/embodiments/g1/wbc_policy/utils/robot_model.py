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
from copy import deepcopy
from typing import List, Optional, Set, Union, TYPE_CHECKING
import yaml
import pinocchio as pin


from isaac_arena.embodiments.g1.wbc_policy.utils.robot_supplemental_info import RobotSupplementalInfo


class RobotModel:
    def __init__(
        self,
        urdf_path,
        asset_path,
        set_floating_base=False,
        supplemental_info: Optional[RobotSupplementalInfo] = None,
    ):
        self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[asset_path],
            root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
        )

        self.is_floating_base_model = set_floating_base

        joints_order_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/loco_manip_g1_joints_order_43dof.yaml")

        with open(joints_order_path, "r") as f:
            self.wbc_g1_joints_order = yaml.safe_load(f)

        self.joint_to_dof_index = {}
        for name in self.wbc_g1_joints_order:
            self.joint_to_dof_index[name] = self.wbc_g1_joints_order[name]

        # Store joint limits only for actual joints (excluding floating base)
        # if set floating base is true and the robot can move in the world
        # then we don't want to impose joint limits for the 7 dofs corresponding
        # to the floating base dofs.
        root_nq = 7 if set_floating_base else 0

        # Set up supplemental info if provided
        self.supplemental_info = supplemental_info
        self.num_dofs_body = len(self.supplemental_info.body_actuated_joints)
        self.num_dofs_hands = len(self.supplemental_info.left_hand_actuated_joints) + len(self.supplemental_info.right_hand_actuated_joints)
        self.lower_joint_limits = np.zeros(self.num_dofs)
        self.upper_joint_limits = np.zeros(self.num_dofs)
        if self.supplemental_info is not None:
            print(f"self.supplemental_info.body_actuated_joints: {self.supplemental_info.body_actuated_joints}")
            # Cache indices for body and hand actuated joints separately
            self._body_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.body_actuated_joints
            ]
            self._left_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.left_hand_actuated_joints
            ]
            self._right_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.right_hand_actuated_joints
            ]
            self._hand_actuated_joint_indices = (
                self._left_hand_actuated_joint_indices + self._right_hand_actuated_joint_indices
            )

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

            # Update joint limits from supplemental info if available
            if (
                hasattr(self.supplemental_info, "joint_limits")
                and self.supplemental_info.joint_limits
            ):
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

        self.initial_body_pose = None

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom of the robot (floating base pose + joints)."""
        # return self.pinocchio_wrapper.model.nq
        return self.num_dofs_body + self.num_dofs_hands

    @property
    def q_default(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        # return self.pinocchio_wrapper.q0
        return self.initial_body_pose

    @property
    def joint_names(self) -> List[str]:
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
                f"Unknown joint name: '{joint_name}'. "
                f"Available joints: {list(self.joint_to_dof_index.keys())}"
            )
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> List[int]:
        """
        Get the indices of body actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> List[int]:
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

    def get_joint_group_indices(self, group_names: Union[str, Set[str]]) -> List[int]:
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


    def get_initial_upper_body_pose(self) -> np.ndarray:
        """
        Get the default upper body pose of the robot.
        """
        if self.initial_body_pose is not None:
            return self.initial_body_pose[self.get_joint_group_indices("upper_body")]

        q = np.zeros(self.num_dofs)
        default_joint_q = self.supplemental_info.default_joint_q
        for joint, sides in default_joint_q.items():
            for side, value in sides.items():
                q[self.dof_index(self.supplemental_info.joint_name_mapping[joint][side])] = value
        self.initial_body_pose = q
        return self.initial_body_pose[self.get_joint_group_indices("upper_body")]

    def set_initial_body_pose(self, q: np.ndarray, q_idx=None) -> None:
        """
        Set the initial body pose of the robot.
        """
        if q_idx is None:
            self.initial_body_pose = q
        else:
            self.initial_body_pose[q_idx] = q

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics to update the pose of every joint and frame
        in the Pinocchio data structures for the given configuration `q`.

        :param q: A numpy array of shape (num_dofs,) representing the robot configuration.
        """
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]} instead.")

        # Apply auto-clip if enabled
        if auto_clip:
            q = self.clip_configuration(q)

        pin.framesForwardKinematics(self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q)

    def clip_configuration(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the configuration to stay within joint limits with a small tolerance.

        :param q: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_clipped = q.copy()

        # Only clip joint positions, not floating base
        root_nq = 7 if self.is_floating_base_model else 0
        q_clipped[root_nq:] = np.clip(
            q[root_nq:], self.lower_joint_limits + margin, self.upper_joint_limits - margin
        )

        return q_clipped



