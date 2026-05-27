# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pathlib
import torch
from typing import Any

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.homie_utils import load_config
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.onnx_utils import OnnxInferenceSession

# LSTM hidden state shape baked into the recurrent student ONNX.
_LSTM_NUM_LAYERS = 1
_LSTM_HIDDEN_SIZE = 256


class G1AgilePolicy(WBCPolicy):
    """G1 robot policy using the recurrent LSTM student exported from
    ``agile/rl_env/tasks/locomotion_height/g1/velocity_height_env_cfg.py``.

    The ONNX has three inputs: a flat ``obs`` vector and the LSTM hidden/cell
    states ``h_in``/``c_in``. ``obs`` is laid out as::

        [velocity_height_commands(4),  # [v_x, v_y, w_z, h]
         base_ang_vel(3),
         projected_gravity(3),
         joint_pos_rel(29) = q - q_default,
         joint_vel_rel(29) * joint_vel_scale,
         last_action(num_actions=12)]

    The 12 outputs are the leg target joint positions (no waist roll/pitch).

    The upstream ONNX is exported with a static batch dimension of 1. For
    ``num_envs > 1``, this adapter creates a cached dynamic-batch copy of the
    model so all environments still run in a single ONNXRuntime call.
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
        model_local_path = retrieve_file_path(model_path, force_download=False)
        model_full_path = parent_dir / model_local_path
        model_full_path = self._get_compatible_model_path(model_full_path)
        self.session = OnnxInferenceSession(str(model_full_path))

        # Build joint index mappings between WBC order and agile ONNX order.
        self._build_joint_mappings()

        # Initialize state.
        self._init_state()

    def _get_compatible_model_path(self, model_path: pathlib.Path) -> pathlib.Path:
        """Return an ONNX path whose batch axes accept ``self.num_envs``."""
        if self.num_envs == 1:
            return model_path

        try:
            import onnx
        except ImportError as exc:
            raise ImportError(
                "G1AgilePolicy requires the 'onnx' package to inspect batch dimensions for num_envs > 1."
            ) from exc

        model = onnx.load(str(model_path))
        if self._model_supports_num_envs(model):
            return model_path

        dynamic_model_path = model_path.with_name(f"{model_path.stem}_dynamic_batch{model_path.suffix}")
        if dynamic_model_path.exists() and dynamic_model_path.stat().st_mtime >= model_path.stat().st_mtime:
            return dynamic_model_path

        def set_dynamic_dim(value_info: Any, axis: int, dim_param: str):
            shape = value_info.type.tensor_type.shape
            if len(shape.dim) <= axis:
                return
            dim = shape.dim[axis]
            dim.ClearField("dim_value")
            dim.dim_param = dim_param

        for value_info in list(model.graph.input) + list(model.graph.output):
            if value_info.name in {"obs", "actions"}:
                set_dynamic_dim(value_info, 0, "batch_size")
            elif value_info.name in {"h_in", "c_in", "h_out", "c_out"}:
                set_dynamic_dim(value_info, 1, "batch_size")

        onnx.save(model, str(dynamic_model_path))
        return dynamic_model_path

    def _model_supports_num_envs(self, model: Any) -> bool:
        """Check whether all known environment batch axes accept ``self.num_envs``."""

        def dim_supports_num_envs(value_info: Any, axis: int) -> bool:
            shape = value_info.type.tensor_type.shape
            if len(shape.dim) <= axis:
                return True
            dim = shape.dim[axis]
            return dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value == self.num_envs

        batch_axes = {
            "obs": 0,
            "actions": 0,
            "h_in": 1,
            "c_in": 1,
            "h_out": 1,
            "c_out": 1,
        }
        for value_info in list(model.graph.input) + list(model.graph.output):
            axis = batch_axes.get(value_info.name)
            if axis is not None and not dim_supports_num_envs(value_info, axis):
                return False
        return True

    def _build_joint_mappings(self):
        """Build index mappings between WBC joint order and agile ONNX joint order."""
        wbc_order = self.robot_model.wbc_g1_joints_order  # {joint_name: wbc_index}

        # Mapping for ONNX input: indices into the WBC-ordered observation to select
        # the 29 body joints in the order the ONNX model expects.
        onnx_input_names = self.config["onnx_input_joint_names"]
        self.wbc_to_agile_input = np.array([wbc_order[name] for name in onnx_input_names])

        # Mapping for ONNX output: for each of the 12 agile output joints, the
        # position in the 15-slot ``lower_body`` array (12 legs + 3 waist; see
        # ``G1SupplementalInfo.joint_groups`` in ``g1_supplemental_info.py``).
        # AGILE only drives the 12 leg slots; the 3 waist slots stay at zero.
        controlled_names = self.config["controlled_joint_names"]
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        self.agile_idx_to_lower_body_idx = np.array(
            [lower_body_indices.index(wbc_order[name]) for name in controlled_names]
        )

        self.num_lower_body = len(lower_body_indices)
        self.num_actions = int(self.config["num_actions"])

    def _init_state(self):
        """Initialize all per-environment state variables."""
        self.observation = None

        # 4-dim command: [v_x, v_y, w_z, h].
        self._cmd_init = np.array(self.config["cmd_init"], dtype=np.float32)
        assert self._cmd_init.shape == (4,), f"cmd_init must be 4-dim, got {self._cmd_init.shape}"
        self.cmd = np.tile(self._cmd_init, (self.num_envs, 1))

        # LSTM feedback state.
        self.h_state = np.zeros((_LSTM_NUM_LAYERS, self.num_envs, _LSTM_HIDDEN_SIZE), dtype=np.float32)
        self.c_state = np.zeros_like(self.h_state)

        # Previous action fed back as part of obs.
        self.last_action = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        self.joint_vel_scale = float(self.config.get("joint_vel_scale", 0.1))

        # Action post-processing (matches agile JointPositionActionCfg):
        #   target_q = clip(raw_action, action_clip) * action_scale + action_offset
        action_clip = self.config.get("action_clip")
        if action_clip is not None:
            self._action_clip_min = float(action_clip[0])
            self._action_clip_max = float(action_clip[1])
        else:
            self._action_clip_min = -np.inf
            self._action_clip_max = np.inf
        action_scale = self.config.get("action_scale", [1.0] * self.num_actions)
        action_offset = self.config.get("action_offset", [0.0] * self.num_actions)
        self._action_scale = np.array(action_scale, dtype=np.float32).reshape(1, -1)
        self._action_offset = np.array(action_offset, dtype=np.float32).reshape(1, -1)
        assert self._action_scale.shape == (
            1,
            self.num_actions,
        ), f"action_scale must have {self.num_actions} entries, got {self._action_scale.shape}"
        assert self._action_offset.shape == (
            1,
            self.num_actions,
        ), f"action_offset must have {self.num_actions} entries, got {self._action_offset.shape}"

    def reset(self, env_ids: torch.Tensor):
        """Reset the policy state for the given environment ids.

        Args:
            env_ids: The environment ids to reset.
        """
        idx = env_ids.cpu().numpy() if isinstance(env_ids, torch.Tensor) else np.asarray(env_ids)
        self.h_state[:, idx, :] = 0.0
        self.c_state[:, idx, :] = 0.0
        self.last_action[idx] = 0.0
        self.cmd[idx] = self._cmd_init

    def set_observation(self, observation: dict[str, Any]):
        """Store the current observation for the next get_action call.

        Args:
            observation: Dictionary containing robot state from prepare_observations().
        """
        self.observation = observation

    def set_goal(self, goal: dict[str, Any]):
        """Set the goal for the policy.

        ``self.cmd`` is a 4-wide buffer ``[v_x, v_y, w_z, h]`` initialized from
        ``cmd_init``. Each key below updates only its own slice, so callers that
        provide one without the other (e.g. a velocity-only navigation override)
        leave the height channel at its previous value rather than silently
        no-op'ing.

        Args:
            goal: Dictionary containing goals. Supported keys:
                - "navigate_cmd": velocity command array of shape (num_envs, 3)
                - "base_height_command": height command of shape (num_envs, 1)
        """
        nav = goal.get("navigate_cmd")
        if nav is not None:
            nav = np.asarray(nav, dtype=np.float32)
            assert nav.shape == (
                self.num_envs,
                3,
            ), f"navigate_cmd must have shape ({self.num_envs}, 3), got {nav.shape}"
            self.cmd[:, :3] = nav

        height = goal.get("base_height_command")
        if height is not None:
            height = np.asarray(height, dtype=np.float32)
            assert height.shape == (
                self.num_envs,
                1,
            ), f"base_height_command must have shape ({self.num_envs}, 1), got {height.shape}"
            self.cmd[:, 3:4] = height

    def get_action(self, time: float | None = None) -> dict[str, Any]:
        """Compute and return the next action based on current observation.

        Returns:
            Dictionary with "body_action" key containing joint position targets
            of shape (num_envs, num_lower_body_joints).
        """
        if self.observation is None:
            raise ValueError("No observation set. Call set_observation() first.")

        obs = self.observation

        # Use root_ang_vel_b (COM frame) to match `mdp.base_ang_vel` from training.
        ang_vel_b = obs["root_ang_vel_b"].astype(np.float32)
        proj_grav = obs["projected_gravity_b"].astype(np.float32)

        # Reorder WBC-ordered q/dq/default_q into the agile training order.
        q_agile = obs["q"][:, self.wbc_to_agile_input].astype(np.float32)
        dq_agile = obs["dq"][:, self.wbc_to_agile_input].astype(np.float32)
        q_default_agile = obs["default_q"][:, self.wbc_to_agile_input].astype(np.float32)

        joint_pos_rel = q_agile - q_default_agile
        joint_vel_rel = dq_agile * self.joint_vel_scale

        flat_obs = np.concatenate(
            [self.cmd, ang_vel_b, proj_grav, joint_pos_rel, joint_vel_rel, self.last_action],
            axis=1,
        )

        result = self.session.run({"obs": flat_obs.astype(np.float32), "h_in": self.h_state, "c_in": self.c_state})

        actions = result["actions"].astype(np.float32)  # (N, num_actions)
        self.h_state = result["h_out"]
        self.c_state = result["c_out"]

        # Feed back the *raw* (pre-processed) action: this matches mdp.last_action
        # which returns env.action_manager.action (the policy output before scale/offset).
        self.last_action = actions

        # Match the agile training JointPositionActionCfg post-processing:
        #   target_q = clip(raw_action, clip_min, clip_max) * scale + offset
        target_q = np.clip(actions, self._action_clip_min, self._action_clip_max)
        target_q = target_q * self._action_scale + self._action_offset

        body_action = np.zeros((self.num_envs, self.num_lower_body), dtype=np.float32)
        body_action[:, self.agile_idx_to_lower_body_idx] = target_q
        return {"body_action": body_action}
