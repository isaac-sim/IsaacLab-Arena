# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import numpy as np
import pathlib
import torch
import urllib.request
from typing import Any

import onnxruntime as ort

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.homie_utils import load_config

_AGILE_MODEL_URL = (
    "https://github.com/nvidia-isaac/WBC-AGILE/raw/main/agile/data/policy/velocity_g1/unitree_g1_velocity_e2e.onnx"
)
_AGILE_MODEL_SHA256 = "8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b"


def _download_agile_model(dest: pathlib.Path) -> None:
    """Download the AGILE ONNX model and verify its SHA256 checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AGILE ONNX model to {dest} ...")
    urllib.request.urlretrieve(_AGILE_MODEL_URL, dest)

    actual = hashlib.sha256(dest.read_bytes()).hexdigest()
    if actual != _AGILE_MODEL_SHA256:
        dest.unlink()
        raise RuntimeError(f"SHA256 mismatch for AGILE model: expected {_AGILE_MODEL_SHA256}, got {actual}")
    print(f"Downloaded and verified AGILE model: {dest}")


class G1AgilePolicy(WBCPolicy):
    """G1 robot policy using the WBC-AGILE end-to-end neural network.

    This policy uses a single ONNX model that takes raw sensor inputs and
    manages observation history internally via feedback connections. The model
    outputs target joint positions along with per-joint Kp/Kd gains for 14
    controlled joints (legs + waist_roll + waist_pitch).
    """

    def __init__(self, robot_model, config_path: str, model_path: str, num_envs: int = 1):
        """Initialize G1AgilePolicy.

        Args:
            robot_model: Robot model containing joint ordering info.
            config_path: Path to policy YAML configuration file (relative to wbc_policy dir).
            model_path: Path to the ONNX model file (relative to wbc_policy dir).
            num_envs: Number of environments.
        """
        parent_dir = pathlib.Path(__file__).parent.parent
        self.config = load_config(str(parent_dir / config_path))
        self.robot_model = robot_model
        self.num_envs = num_envs

        # Download ONNX model on first use if not already present
        model_full_path = parent_dir / model_path
        if not model_full_path.exists():
            _download_agile_model(model_full_path)
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
        self.wbc_to_agile_input = [wbc_order[name] for name in onnx_input_names]

        # Mapping for ONNX output: for each of the 14 agile output joints, the
        # position in the 15-element lower_body array to write to.
        controlled_names = self.config["controlled_joint_names"]
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        self.agile_output_to_lower_body = []
        for name in controlled_names:
            wbc_idx = wbc_order[name]
            lb_pos = lower_body_indices.index(wbc_idx)
            self.agile_output_to_lower_body.append(lb_pos)

        self.num_lower_body = len(lower_body_indices)

    def _init_state(self):
        """Initialize all per-environment state variables."""
        self.observation = None
        self.use_policy_action = True
        self.cmd = np.tile(self.config["cmd_init"], (self.num_envs, 1))

        # Per-environment ONNX feedback state. Each entry is shaped for batch=1
        # as the ONNX model expects, matching the input tensor shapes from the YAML.
        self.states = [self._make_zero_state() for _ in range(self.num_envs)]

    def _make_zero_state(self) -> dict[str, np.ndarray]:
        """Create a zeroed feedback state dict for one environment."""
        num_actions = self.config["num_actions"]
        # History length (5) and angular velocity dims (3) are intrinsic to the ONNX model architecture.
        return {
            "last_actions": np.zeros((1, num_actions), dtype=np.float32),
            "base_ang_vel_history": np.zeros((1, 5, 3), dtype=np.float32),
            "projected_gravity_history": np.zeros((1, 5, 3), dtype=np.float32),
            "velocity_commands_history": np.zeros((1, 5, 3), dtype=np.float32),
            "controlled_joint_pos_history": np.zeros((1, 5, num_actions), dtype=np.float32),
            "controlled_joint_vel_history": np.zeros((1, 5, num_actions), dtype=np.float32),
            "actions_history": np.zeros((1, 5, num_actions), dtype=np.float32),
        }

    def reset(self, env_ids: torch.Tensor):
        """Reset the policy state for the given environment ids.

        Args:
            env_ids: The environment ids to reset.
        """
        for env_id in env_ids:
            idx = int(env_id)
            self.states[idx] = self._make_zero_state()
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
                - "toggle_policy_action": bool to toggle policy action on/off
        """
        if "toggle_policy_action" in goal:
            if goal["toggle_policy_action"]:
                self.use_policy_action = not self.use_policy_action

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
        body_action = np.zeros((self.num_envs, self.num_lower_body), dtype=np.float32)

        for env_idx in range(self.num_envs):
            # Build ONNX inputs for this environment
            ort_inputs = self._build_onnx_inputs(obs, env_idx)

            # Run inference
            outputs = self.session.run(self.output_names, ort_inputs)
            result = dict(zip(self.output_names, outputs))

            # Extract action joint positions (shape [1, 14])
            action_joint_pos = result["action_joint_pos"]

            # Update feedback state for next step
            state = self.states[env_idx]
            state["last_actions"] = result["last_actions_out"]
            state["base_ang_vel_history"] = result["base_ang_vel_history_out"]
            state["projected_gravity_history"] = result["projected_gravity_history_out"]
            state["velocity_commands_history"] = result["velocity_commands_history_out"]
            state["controlled_joint_pos_history"] = result["controlled_joint_pos_history_out"]
            state["controlled_joint_vel_history"] = result["controlled_joint_vel_history_out"]
            state["actions_history"] = result["actions_history_out"]

            # Map 14 agile output joints to the 15-joint lower_body array.
            # waist_yaw (not controlled by agile) stays at 0.0.
            if self.use_policy_action:
                for agile_idx, lb_idx in enumerate(self.agile_output_to_lower_body):
                    body_action[env_idx, lb_idx] = action_joint_pos[0, agile_idx]
            else:
                body_action[env_idx] = obs["q"][env_idx, : self.num_lower_body]

        return {"body_action": body_action}

    def _build_onnx_inputs(self, obs: dict[str, Any], env_idx: int) -> dict[str, np.ndarray]:
        """Build the ONNX input dict for a single environment.

        Args:
            obs: Observation dictionary from prepare_observations().
            env_idx: Environment index.

        Returns:
            Dictionary mapping ONNX input names to numpy arrays.
        """
        # Quaternion (w, x, y, z) from floating base pose
        root_link_quat_w = obs["floating_base_pose"][env_idx : env_idx + 1, 3:7].astype(np.float32)

        # Angular velocity in body frame
        root_ang_vel_b = obs["floating_base_vel"][env_idx : env_idx + 1, 3:6].astype(np.float32)

        # Velocity commands
        velocity_commands = self.cmd[env_idx : env_idx + 1].astype(np.float32)

        # Joint positions and velocities: select 29 body joints and reorder to agile order
        joint_pos = obs["q"][env_idx : env_idx + 1, self.wbc_to_agile_input].astype(np.float32)
        joint_vel = obs["dq"][env_idx : env_idx + 1, self.wbc_to_agile_input].astype(np.float32)

        state = self.states[env_idx]

        return {
            "root_link_quat_w": root_link_quat_w,
            "root_ang_vel_b": root_ang_vel_b,
            "velocity_commands": velocity_commands,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "last_actions": state["last_actions"],
            "base_ang_vel_history": state["base_ang_vel_history"],
            "projected_gravity_history": state["projected_gravity_history"],
            "velocity_commands_history": state["velocity_commands_history"],
            "controlled_joint_pos_history": state["controlled_joint_pos_history"],
            "controlled_joint_vel_history": state["controlled_joint_vel_history"],
            "actions_history": state["actions_history"],
        }
