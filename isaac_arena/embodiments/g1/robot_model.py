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
from copy import deepcopy

import pinocchio as pin

from isaac_arena.embodiments.g1.g1_supplemental_info import G1SupplementalInfo


class RobotModel:
    def __init__(
        self,
        urdf_path,
        asset_path,
        set_floating_base=False,
        supplemental_info: G1SupplementalInfo | None = None,
    ):
        self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[asset_path],
            root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
        )

        self.is_floating_base_model = set_floating_base

        joints_order_path = os.path.join(os.path.dirname(__file__), "config/loco_manip_g1_joints_order_43dof.yaml")

        with open(joints_order_path) as f:
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
        print(f"self.supplemental_info: {self.supplemental_info}")
        self.num_dofs_body = len(self.supplemental_info.body_actuated_joints)
        self.num_dofs_hands = len(self.supplemental_info.left_hand_actuated_joints) + len(
            self.supplemental_info.right_hand_actuated_joints
        )
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
            if hasattr(self.supplemental_info, "joint_limits") and self.supplemental_info.joint_limits:
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom of the robot (floating base pose + joints)."""
        return self.num_dofs_body + self.num_dofs_hands

    @property
    def q_default(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        return self.pinocchio_wrapper.q0

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

    def frame_placement(self, frame_name: str) -> pin.SE3:
        """
        Returns the SE3 transform of the specified frame in the world coordinate system.
        Note: make sure cache_forward_kinematics() has been previously called.

        :param frame_name: Name of the frame, e.g. "link_elbow_frame", "hand_imu_frame", etc.
        :return: A pin.SE3 object representing the pose of the frame.
        """
        model = self.pinocchio_wrapper.model
        data = self.pinocchio_wrapper.data

        frame_id = model.getFrameId(frame_name)
        if frame_id < 0 or frame_id >= len(model.frames):
            valid_frames = [f.name for f in model.frames]
            raise ValueError(f"Unknown frame '{frame_name}'. Valid frames: {valid_frames}")

        # Pinocchio's data.oMf[frame_id] is a pin.SE3.
        return data.oMf[frame_id].copy()

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
        q_clipped[root_nq:] = np.clip(q[root_nq:], self.lower_joint_limits + margin, self.upper_joint_limits - margin)

        return q_clipped

    def reset_forward_kinematics(self) -> None:
        """
        Reset the forward kinematics to the initial configuration.
        """
        self.cache_forward_kinematics(self.q_default)


class ReducedRobotModel:
    """
    A class that creates a reduced order robot model by fixing certain joints.
    This class maintains a mapping between the reduced state space and the full state space.
    """

    def __init__(
        self,
        full_robot_model: RobotModel,
        fixed_joints: list[str],
        fixed_values: list[float] | None = None,
    ):
        """
        Create a reduced order robot model by fixing specified joints.

        :param full_robot_model: The original robot model
        :param fixed_joints: List of joint names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        """
        self.full_robot = full_robot_model
        self.supplemental_info = full_robot_model.supplemental_info

        # If fixed_values is None, use q0 from the full robot model
        if fixed_values is None:
            fixed_values = []
            for joint_name in fixed_joints:
                full_idx = full_robot_model.dof_index(joint_name)
                fixed_values.append(full_robot_model.pinocchio_wrapper.q0[full_idx])
        elif len(fixed_joints) != len(fixed_values):
            raise ValueError("fixed_joints and fixed_values must have the same length")

        # Store fixed joints and their values
        self.fixed_joints = fixed_joints
        self.fixed_values = fixed_values

        # Create mapping between reduced and full state spaces
        self.reduced_to_full = []
        self.full_to_reduced = {}

        # Initialize with floating base indices if present
        if full_robot_model.is_floating_base_model:
            self.reduced_to_full.extend(range(7))  # Floating base indices
            for i in range(7):
                self.full_to_reduced[i] = i

        # Add active joint indices
        for joint_name in full_robot_model.joint_names:
            if joint_name not in fixed_joints:
                full_idx = full_robot_model.dof_index(joint_name)
                reduced_idx = len(self.reduced_to_full)
                self.reduced_to_full.append(full_idx)
                self.full_to_reduced[full_idx] = reduced_idx

        # Create a reduced Pinocchio model using buildReducedModel
        # First, get the list of joint IDs to lock
        locked_joint_ids = []
        for joint_name in fixed_joints:
            joint_id = full_robot_model.pinocchio_wrapper.model.getJointId(joint_name)
            if (full_robot_model.is_floating_base_model and joint_id > 1) or (
                not full_robot_model.is_floating_base_model and joint_id > 0
            ):
                locked_joint_ids.append(joint_id)

        # First build the reduced kinematic model
        reduced_model = pin.buildReducedModel(
            full_robot_model.pinocchio_wrapper.model,
            locked_joint_ids,
            full_robot_model.pinocchio_wrapper.q0,
        )

        # Then build the reduced geometry models using the reduced kinematic model
        self.pinocchio_wrapper = pin.RobotWrapper(
            model=reduced_model,
        )

        # Create joint to dof index mapping
        self.joint_to_dof_index = {}
        # Assume we only have single-dof joints
        # First two names correspond to universe and floating base joints
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if self.full_robot.is_floating_base_model
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q

        # Initialize joint limits
        root_nq = 7 if self.full_robot.is_floating_base_model else 0
        self.lower_joint_limits = deepcopy(self.pinocchio_wrapper.model.lowerPositionLimit[root_nq:])
        self.upper_joint_limits = deepcopy(self.pinocchio_wrapper.model.upperPositionLimit[root_nq:])

        # Update joint limits from supplemental info if available
        if self.supplemental_info is not None:
            for joint_name, limits in self.supplemental_info.joint_limits.items():
                if joint_name in self.joint_to_dof_index:
                    idx = self.joint_to_dof_index[joint_name] - root_nq
                    self.lower_joint_limits[idx] = limits[0]
                    self.upper_joint_limits[idx] = limits[1]

        # Cache indices for body and hand actuated joints in reduced space
        if self.supplemental_info is not None:
            # Get full indices for body and hand actuated joints
            full_body_indices = full_robot_model.get_body_actuated_joint_indices()
            full_hand_indices = full_robot_model.get_hand_actuated_joint_indices("both")

            # Map to reduced indices
            self._body_actuated_joint_indices = []
            for idx in full_body_indices:
                if idx in self.full_to_reduced:
                    self._body_actuated_joint_indices.append(self.full_to_reduced[idx])

            self._hand_actuated_joint_indices = []
            for idx in full_hand_indices:
                if idx in self.full_to_reduced:
                    self._hand_actuated_joint_indices.append(self.full_to_reduced[idx])

            # Cache indices for joint groups in reduced space
            self._joint_group_indices = {}
            for group_name in self.supplemental_info.joint_groups:
                full_indices = full_robot_model.get_joint_group_indices(group_name)
                reduced_indices = []
                for idx in full_indices:
                    if idx in self.full_to_reduced:
                        reduced_indices.append(self.full_to_reduced[idx])
                self._joint_group_indices[group_name] = sorted(set(reduced_indices))

    @classmethod
    def from_fixed_groups(
        cls,
        full_robot_model: RobotModel,
        fixed_group_names: list[str],
        fixed_values: list[float] | None = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in specified groups.

        :param full_robot_model: The original robot model
        :param fixed_group_names: List of joint group names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        if full_robot_model.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Get all joints in the groups, including those from subgroups
        fixed_joints = set()  # Use a set to avoid duplicates

        for group_name in fixed_group_names:
            if group_name not in full_robot_model.supplemental_info.joint_groups:
                raise ValueError(f"Unknown joint group: {group_name}")

            group_info = full_robot_model.supplemental_info.joint_groups[group_name]

            # Add direct joints
            fixed_joints.update(group_info["joints"])

            # Add joints from subgroups
            for subgroup_name in group_info["groups"]:
                subgroup_joints = full_robot_model.get_joint_group_indices(subgroup_name)
                fixed_joints.update([full_robot_model.joint_names[idx] for idx in subgroup_joints])

        # Convert set back to list for compatibility with the original constructor
        return cls(full_robot_model, list(fixed_joints), fixed_values)

    @classmethod
    def from_fixed_group(
        cls,
        full_robot_model: RobotModel,
        fixed_group_name: str,
        fixed_values: list[float] | None = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in a specified group.
        This is a convenience method that calls from_fixed_groups with a single group.

        :param full_robot_model: The original robot model
        :param fixed_group_name: Name of the joint group to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_fixed_groups(full_robot_model, [fixed_group_name], fixed_values)

    @classmethod
    def from_active_group(
        cls,
        full_robot_model: RobotModel,
        active_group_name: str,
        fixed_values: list[float] | None = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified group.
        This is a convenience method that calls from_active_groups with a single group.

        :param full_robot_model: The original robot model
        :param active_group_name: Name of the joint group to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_active_groups(full_robot_model, [active_group_name], fixed_values)

    @classmethod
    def from_active_groups(
        cls,
        full_robot_model: RobotModel,
        active_group_names: list[str],
        fixed_values: list[float] | None = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified groups.
        This is useful when you want to keep multiple groups active and fix everything else.

        :param full_robot_model: The original robot model
        :param active_group_names: List of joint group names to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        if full_robot_model.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Get all joints in the active groups, including those from subgroups
        active_joints = set()

        def add_group_joints(group_name: str):
            if group_name not in full_robot_model.supplemental_info.joint_groups:
                raise ValueError(f"Unknown joint group: {group_name}")

            group_info = full_robot_model.supplemental_info.joint_groups[group_name]

            # Add direct joints
            if "joints" in group_info:
                active_joints.update(group_info["joints"])

            # Add joints from subgroups
            if "groups" in group_info:
                for subgroup_name in group_info["groups"]:
                    add_group_joints(subgroup_name)

        for group_name in active_group_names:
            add_group_joints(group_name)

        # Get all joints from the model
        all_joints = set(full_robot_model.joint_names)

        # The fixed joints are all joints minus the active joints
        fixed_joints = list(all_joints - active_joints)

        return cls(full_robot_model, fixed_joints, fixed_values)

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom in the reduced model."""
        return self.pinocchio_wrapper.model.nq

    @property
    def q_default(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        return self.pinocchio_wrapper.q0

    @property
    def joint_names(self) -> list[str]:
        """Get the names of the active joints in the reduced model."""
        return [name for name in self.full_robot.joint_names if name not in self.fixed_joints]

    def get_joint_group_indices(self, group_names: str | set[str]) -> list[int]:
        """
        Get the indices of joints in one or more groups in the reduced configuration.
        Requires supplemental_info to be provided.
        The returned indices are sorted in ascending order, so that the joint ordering
        of the reduced model is preserved.

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

    def get_body_actuated_joint_indices(self) -> list[int]:
        """
        Get the indices of body actuated joints in the reduced configuration.
        Requires supplemental_info to be provided.
        """
        if not hasattr(self, "_body_actuated_joint_indices"):
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> list[int]:
        """
        Get the indices of hand actuated joints in the reduced configuration.
        Requires supplemental_info to be provided.

        Args:
            side: String specifying which hand to get indices for ('left', 'right', or 'both')
        """
        if not hasattr(self, "_hand_actuated_joint_indices"):
            raise ValueError("supplemental_info must be provided to use this method")

        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_body_actuated_joints(self, q_reduced: np.ndarray) -> np.ndarray:
        """
        Get the configuration of body actuated joints from a reduced configuration.

        :param q_reduced: Configuration in reduced space
        :return: Configuration of body actuated joints
        """
        indices = self.get_body_actuated_joint_indices()
        return q_reduced[indices]

    def get_hand_actuated_joints(self, q_reduced: np.ndarray, side: str = "both") -> np.ndarray:
        """
        Get the configuration of hand actuated joints from a reduced configuration.

        Args:
            q_reduced: Configuration in reduced space
            side: String specifying which hand to get joints for ('left', 'right', or 'both')
        """
        indices = self.get_hand_actuated_joint_indices(side)
        return q_reduced[indices]

    def get_configuration_from_actuated_joints(
        self,
        body_actuated_joint_values: np.ndarray,
        hand_actuated_joint_values: np.ndarray | None = None,
        left_hand_actuated_joint_values: np.ndarray | None = None,
        right_hand_actuated_joint_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get the reduced configuration from the body and hand actuated joint configurations.
        Can specify either both hands together or left and right hands separately.

        Args:
            body_actuated_joint_values: Configuration of body actuated joints
            hand_actuated_joint_values: Configuration of both hands' actuated joints (optional)
            left_hand_actuated_joint_values: Configuration of left hand actuated joints (optional)
            right_hand_actuated_joint_values: Configuration of right hand actuated joints (optional)

        Returns:
            Configuration in reduced space
        """
        # Start with the initial configuration from the full robot model
        q_reduced = self.full_to_reduced_configuration(self.full_robot.pinocchio_wrapper.q0)

        # Set body actuated joints
        body_indices = self.get_body_actuated_joint_indices()
        q_reduced[body_indices] = body_actuated_joint_values

        # Handle hand configurations
        if hand_actuated_joint_values is not None:
            # Use combined hand configuration
            q_reduced[self.get_hand_actuated_joint_indices("both")] = hand_actuated_joint_values
        else:
            # Use separate hand configurations
            if left_hand_actuated_joint_values is not None:
                q_reduced[self.get_hand_actuated_joint_indices("left")] = left_hand_actuated_joint_values
            if right_hand_actuated_joint_values is not None:
                q_reduced[self.get_hand_actuated_joint_indices("right")] = right_hand_actuated_joint_values

        return q_reduced

    def reduced_to_full_configuration(self, q_reduced: np.ndarray) -> np.ndarray:
        """
        Convert a reduced configuration to the full configuration space.

        :param q_reduced: Configuration in reduced space
        :return: Configuration in full space with fixed joints set to their fixed values
        """
        if q_reduced.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q_reduced of length {self.num_dofs}, got {q_reduced.shape[0]} instead")

        q_full = np.zeros(self.full_robot.num_dofs)

        # Set active joints
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            q_full[full_idx] = q_reduced[reduced_idx]

        # Set fixed joints
        for joint_name, value in zip(self.fixed_joints, self.fixed_values):
            full_idx = self.full_robot.dof_index(joint_name)
            q_full[full_idx] = value

        return q_full

    def full_to_reduced_configuration(self, q_full: np.ndarray) -> np.ndarray:
        """
        Convert a full configuration to the reduced configuration space.

        :param q_full: Configuration in full space
        :return: Configuration in reduced space
        """
        if q_full.shape[0] != self.full_robot.num_dofs:
            raise ValueError(f"Expected q_full of length {self.full_robot.num_dofs}, got {q_full.shape[0]} instead")

        q_reduced = np.zeros(self.num_dofs)

        # Copy active joints
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            q_reduced[reduced_idx] = q_full[full_idx]

        return q_reduced

    def cache_forward_kinematics(self, q_reduced: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics using the reduced configuration.

        :param q_reduced: Configuration in reduced space
        """
        # First update the full robot's forward kinematics
        q_full = self.reduced_to_full_configuration(q_reduced)
        self.full_robot.cache_forward_kinematics(q_full, auto_clip)

        # Then update the reduced model's forward kinematics
        pin.framesForwardKinematics(self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q_reduced)

    def clip_configuration(self, q_reduced: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the reduced configuration to stay within joint limits with a small tolerance.

        :param q_reduced: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_full = self.reduced_to_full_configuration(q_reduced)
        q_full_clipped = self.full_robot.clip_configuration(q_full, margin)
        return self.full_to_reduced_configuration(q_full_clipped)

    def frame_placement(self, frame_name: str) -> pin.SE3:
        """
        Get the placement of a frame in the reduced model.
        Note: make sure cache_forward_kinematics() has been previously called.

        :param frame_name: Name of the frame
        :return: A pin.SE3 object representing the pose of the frame
        """
        return self.full_robot.frame_placement(frame_name)

    def reset_forward_kinematics(self):
        self.full_robot.reset_forward_kinematics()
