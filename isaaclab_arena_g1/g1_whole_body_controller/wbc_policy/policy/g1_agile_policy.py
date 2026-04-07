# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib
import torch
from typing import Any

import onnxruntime as ort
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.homie_utils import load_config

# ONNX feedback state keys and their per-environment shapes (excluding batch dim).
_STATE_KEYS_AND_SHAPES: dict[str, tuple[int, ...]] = {
    "last_actions": (14,),
    "base_ang_vel_history": (5, 3),
    "projected_gravity_history": (5, 3),
    "velocity_commands_history": (5, 3),
    "controlled_joint_pos_history": (5, 14),
    "controlled_joint_vel_history": (5, 14),
    "actions_history": (5, 14),
}


class G1AgilePolicy(WBCPolicy):
    """G1 robot policy using the WBC-AGILE end-to-end neural network.

    This policy uses a single ONNX model that takes raw sensor inputs and
    manages observation history internally via feedback connections. The model
    outputs target joint positions along with per-joint Kp/Kd gains for 14
    controlled joints (legs + waist_roll + waist_pitch).
    """

    def __init__(self, robot_model: RobotModel, config_path: str, model_path: str, num_envs: int = 1):
        """Initialize G1AgilePolicy.

        Args:
            robot_model: Robot model containing joint ordering info.
            config_path: Path to policy YAML configuration file (relative to wbc_policy dir).
            model_path: Path to the ONNX model file. Can be an S3/Nucleus URL
                (resolved and cached by retrieve_file_path) or a local path.
            num_envs: Number of environments.
        """
        parent_dir = pathlib.Path(__file__).parent.parent
        self.config = load_config(str(parent_dir / config_path))
        self.robot_model = robot_model
        self.num_envs = num_envs

        # Resolve model path via OV asset API (handles S3 download + local caching).
        # Same pattern as G1HomiePolicyV2: retrieve_file_path returns a local path
        # (absolute for S3 cache, relative for local files), then join with parent_dir.
        model_local_path = retrieve_file_path(model_path, force_download=True)
        model_full_path = parent_dir / model_local_path
        self.session = ort.InferenceSession(str(model_full_path))
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"Successfully loaded ONNX policy from {model_full_path}")

        # Build joint index mappings between WBC order and agile ONNX order
        self._build_joint_mappings()

        # Initialize state
        self._init_state()

    def _build_joint_mappings(self):
        """Build index mappings between WBC joint order and agile ONNX joint order."""
        wbc_order = self.robot_model.wbc_g1_joints_order  # {joint_name: wbc_index}

        # Mapping for ONNX input: indices into the WBC-ordered observation to select
        # the 29 body joints in the order the ONNX model expects.
        onnx_input_names = self.config["onnx_input_joint_names"]
        self.wbc_to_agile_input = np.array([wbc_order[name] for name in onnx_input_names])

        # Mapping for ONNX output: for each of the 14 agile output joints, the
        # position in the 15-element lower_body array to write to.
        controlled_names = self.config["controlled_joint_names"]
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        self.agile_idx_to_lower_body_idx = np.array(
            [lower_body_indices.index(wbc_order[name]) for name in controlled_names]
        )

        self.num_lower_body = len(lower_body_indices)

    def _init_state(self):
        """Initialize all per-environment state variables."""
        self.observation = None
        self.cmd = np.tile(self.config["cmd_init"], (self.num_envs, 1))

        # Batched ONNX feedback state: each array has shape (num_envs, ...).
        self.state = {
            key: np.zeros((self.num_envs, *shape), dtype=np.float32) for key, shape in _STATE_KEYS_AND_SHAPES.items()
        }

    def reset(self, env_ids: torch.Tensor):
        """Reset the policy state for the given environment ids.

        Args:
            env_ids: The environment ids to reset.
        """
        idx = env_ids.cpu().numpy()
        for key, shape in _STATE_KEYS_AND_SHAPES.items():
            self.state[key][idx] = np.zeros(shape, dtype=np.float32)
        self.cmd[idx] = self.config["cmd_init"]

    def set_observation(self, observation: dict[str, Any]):
        """Store the current observation for the next get_action call.

        Args:
            observation: Dictionary containing robot state from prepare_observations().
        """
        self.observation = observation

    def set_goal(self, goal: dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing goals. Supported keys:
                - "navigate_cmd": velocity command array of shape (num_envs, 3)
        """
        if "navigate_cmd" in goal:
            self.cmd = goal["navigate_cmd"]

    def get_action(self, time: float | None = None) -> dict[str, Any]:
        """Compute and return the next action based on current observation.

        Returns:
            Dictionary with "body_action" key containing joint position targets
            of shape (num_envs, num_lower_body_joints).
        """
        if self.observation is None:
            raise ValueError("No observation set. Call set_observation() first.")

        obs = self.observation

        # Build batched ONNX inputs (all envs at once)
        ort_inputs = {
            "root_link_quat_w": obs["floating_base_pose"][:, 3:7].astype(np.float32),
            "root_ang_vel_b": obs["floating_base_vel"][:, 3:6].astype(np.float32),
            "velocity_commands": self.cmd.astype(np.float32),
            "joint_pos": obs["q"][:, self.wbc_to_agile_input].astype(np.float32),
            "joint_vel": obs["dq"][:, self.wbc_to_agile_input].astype(np.float32),
            **{key: self.state[key] for key in _STATE_KEYS_AND_SHAPES},
        }

        # Run batched inference
        outputs = self.session.run(self.output_names, ort_inputs)
        result = dict(zip(self.output_names, outputs))

        # Update feedback state for next step
        for key in _STATE_KEYS_AND_SHAPES:
            self.state[key] = result[f"{key}_out"]

        # Map 14 agile output joints to the 15-joint lower_body array.
        # waist_yaw (not controlled by agile) stays at 0.0.
        body_action = np.zeros((self.num_envs, self.num_lower_body), dtype=np.float32)
        body_action[:, self.agile_idx_to_lower_body_idx] = result["action_joint_pos"]

        return {"body_action": body_action}
