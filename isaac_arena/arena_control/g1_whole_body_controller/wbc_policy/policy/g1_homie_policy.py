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

import collections
import numpy as np
import pathlib
import torch
from collections.abc import Callable
from typing import Any

import onnxruntime as ort
from isaaclab.utils.assets import retrieve_file_path

from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy
from isaac_arena.arena_control.g1_whole_body_controller.wbc_policy.utils.homie_utils import (
    get_gravity_orientation,
    load_config,
)


class G1HomiePolicyV2(WBCPolicy):
    """Simple G1 robot policy using OpenHomie trained neural network."""

    def __init__(self, robot_model, config_path: str, model_path: str, num_envs: int = 1):
        """Initialize G1HomiePolicy.

        Args:
            robot_model: Robot model containing supplemental info
            config_path: Path to policy YAML configuration file
            model_path: Path to policy ONNX model file
            num_envs: Number of environments
        """
        parent_dir = pathlib.Path(__file__).parent.parent
        self.config = load_config(str(parent_dir / config_path))
        self.robot_model = robot_model

        model_path_1, model_path_2 = model_path.split(",")
        model_path_1_local = retrieve_file_path(model_path_1, force_download=True)
        model_path_2_local = retrieve_file_path(model_path_2, force_download=True)

        self.policy_1 = self.load_onnx_policy(str(parent_dir / model_path_1_local))
        self.policy_2 = self.load_onnx_policy(str(parent_dir / model_path_2_local))

        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros((num_envs, self.config["num_obs"]), dtype=np.float32)

        # Initialize state variables
        self.use_policy_action = True
        self.action = np.zeros((num_envs, self.config["num_actions"]), dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]
        self.gait_indices = torch.zeros((num_envs, 1), dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor):
        """Reset the policy.

        Args:
            env_ids: The environment ids to reset
        """
        num_envs = env_ids.shape[0]
        self.gait_indices = torch.zeros((num_envs, 1), dtype=torch.float32)
        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros((num_envs, self.config["num_obs"]), dtype=np.float32)

        # Initialize state variables
        self.use_policy_action = True
        self.action = np.zeros((num_envs, self.config["num_actions"]), dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]

    def load_onnx_policy(self, model_path: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Load the ONNX policy from the model path.

        Args:
            model_path: The path to the ONNX policy model

        Returns:
            The ONNX policy model runnable for forward pass.
        """
        model = ort.InferenceSession(model_path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        print(f"Successfully loaded ONNX policy from {model_path}")

        return run_inference

    def compute_observation(self, observation: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Compute the observation vector from current state"""
        # Get body joint indices (excluding waist roll and pitch)
        self.gait_indices = torch.remainder(self.gait_indices + 0.02 * self.freq_cmd, 1.0)
        durations = torch.full_like(self.gait_indices, 0.5)
        phases = 0.5
        foot_indices = [
            self.gait_indices + phases,  # FL
            self.gait_indices,  # FR
        ]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0)
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (0.5 / (1 - durations[swing]))

        self.clock_inputs = torch.stack([torch.sin(2 * np.pi * fi) for fi in foot_indices], dim=1)

        body_indices = self.robot_model.get_joint_group_indices("body")
        body_indices = [idx for idx in body_indices]

        n_joints = len(body_indices)

        # Extract joint data
        num_envs = observation["q"].shape[0]
        qj = observation["q"][:, body_indices].copy()
        dqj = observation["dq"][:, body_indices].copy()

        # Extract floating base data
        quat = observation["floating_base_pose"][:, 3:7].copy()  # quaternion
        omega = observation["floating_base_vel"][:, 3:6].copy()  # angular velocity

        # Handle default angles padding
        if len(self.config["default_angles"]) < n_joints:
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(self.config["default_angles"])] = self.config["default_angles"]
        else:
            padded_defaults = self.config["default_angles"][:n_joints]

        # Scale the values
        qj_scaled = (qj - padded_defaults) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config["ang_vel_scale"]

        # Calculate single observation dimension
        # single_obs_dim = 86
        single_obs_dim = 3 + 1 + 3 + 3 + 3 + n_joints + n_joints + 15

        # Create single observation

        single_obs = np.zeros((num_envs, single_obs_dim), dtype=np.float32)
        single_obs[:, 0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[:, 3:4] = np.array([self.height_cmd])
        single_obs[:, 4:7] = np.stack([self.roll_cmd, self.pitch_cmd, self.yaw_cmd], axis=1)
        single_obs[:, 7:10] = omega_scaled
        single_obs[:, 10:13] = gravity_orientation.T
        single_obs[:, 13 : 13 + n_joints] = qj_scaled
        single_obs[:, 13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[:, 13 + 2 * n_joints : 13 + 2 * n_joints + 15] = self.action

        return single_obs, single_obs_dim

    def set_observation(self, observation: dict[str, Any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing single observation from current state
                        Should include 'obs' key with current single observation
        """

        # Extract the single observation
        self.observation = observation
        single_obs, single_obs_dim = self.compute_observation(observation)

        # Add current observation to history
        self.obs_history.append(single_obs)

        # Fill history with zeros if not enough observations yet
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))

        # Construct full observation with history
        single_obs_dim = single_obs.shape[1]
        for i, hist_obs in enumerate(self.obs_history):

            start_idx = i * single_obs_dim
            end_idx = start_idx + single_obs_dim
            self.obs_buffer[:, start_idx:end_idx] = hist_obs

        # Convert to tensor for policy
        self.obs_tensor = torch.from_numpy(self.obs_buffer)

        assert self.obs_tensor.shape[1] == self.config["num_obs"]

    def set_goal(self, goal: dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing the goal for the policy
        """

        if "toggle_policy_action" in goal:
            if goal["toggle_policy_action"]:
                self.use_policy_action = not self.use_policy_action

        if "navigate_cmd" in goal:
            self.cmd = goal["navigate_cmd"]

        if "base_height_command" in goal:
            self.height_cmd = (
                goal["base_height_command"][0]
                if isinstance(goal["base_height_command"], list)
                else goal["base_height_command"]
            )

        if "torso_orientation_rpy_cmd" in goal:
            self.roll_cmd = goal["torso_orientation_rpy_cmd"][:, 0]
            self.pitch_cmd = goal["torso_orientation_rpy_cmd"][:, 1]
            self.yaw_cmd = goal["torso_orientation_rpy_cmd"][:, 2]

    def get_action(self) -> dict[str, Any]:
        """Compute and return the next action based on current observation.

        Args:
            time: Optional "monotonic time" for time-dependent policies (unused)

        Returns:
            Dictionary containing the action to be executed
        """
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        # Run policy inference
        with torch.no_grad():
            # Select appropriate policy based on command magnitude
            if np.linalg.norm(self.cmd) < 0.05:
                # Use standing policy for small commands
                policy = self.policy_1
            else:
                # Use walking policy for movement commands
                policy = self.policy_2

            self.action = policy(self.obs_tensor).detach().numpy()

        # Transform action to target_dof_pos
        assert self.use_policy_action
        if self.use_policy_action:
            cmd_q = self.action * self.config["action_scale"] + self.config["default_angles"]
        else:
            cmd_q = self.observation["q"][self.robot_model.get_joint_group_indices("lower_body")]
        # Only produce target joint positions from WBC, no kinematics nor dynamics

        return {"body_action": cmd_q}
