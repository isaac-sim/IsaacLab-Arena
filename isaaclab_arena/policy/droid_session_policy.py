# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID joint-action control through a local controller session."""

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import warp as wp

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.session_exchange import SessionExchange
from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter

_DROID_ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]


@dataclass
class DroidSessionPolicyCfg(PolicyCfg):
    """Configure a DROID controller that responds through local files."""

    controller_prompt_path: str = ""
    """Markdown controller instructions, relative to the runtime working directory."""

    action_chunk_length: int = 15
    """Number of consecutive control steps requested from the controller."""

    response_timeout_s: float = 600.0
    """Wall-clock time to wait for one response while physics is paused."""

    def __post_init__(self) -> None:
        assert self.controller_prompt_path, "controller_prompt_path must name a controller prompt"
        assert type(self.action_chunk_length) is int and self.action_chunk_length > 0
        assert math.isfinite(self.response_timeout_s) and self.response_timeout_s > 0


@register_policy
class DroidSessionPolicy(PolicyBase[DroidSessionPolicyCfg]):
    """Return joint actions supplied by a controller session to the Experiment Runner."""

    name = "droid_session"

    def __init__(self, config: DroidSessionPolicyCfg) -> None:
        super().__init__(config)
        self._adapter = Pi0DroidAdapter()
        self._exchange: SessionExchange | None = None
        self._scheduler: ActionChunkScheduler | None = None
        self._joint_limits: np.ndarray | None = None
        self._episode_index: int | None = None
        self._environment: gym.Env | None = None
        self._closed = False
        self.task_description: str | None = None

    def set_output_directory(self, output_directory: Path) -> None:
        """Create an isolated exchange and snapshot its configuration and prompt."""
        assert self._exchange is None and not self._closed, "Policy output directory is already assigned"
        controller_prompt = Path(self.config.controller_prompt_path).read_text(encoding="utf-8")
        assert controller_prompt.strip(), "Controller prompt must not be empty"
        self._exchange = SessionExchange(
            output_directory=output_directory,
            controller_prompt=controller_prompt,
            policy_configuration={"type": f"{type(self).__module__}.{type(self).__name__}", **asdict(self.config)},
            response_timeout_s=self.config.response_timeout_s,
        )

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return one action, requesting a new chunk only when the buffer is empty."""
        assert not self._closed, "DroidSessionPolicy is closed"
        assert self._exchange is not None, "Run DroidSessionPolicy through the Experiment Runner to assign its output"
        try:
            assert self.task_description, "DroidSessionPolicy requires a task language instruction"
            if self._scheduler is None:
                self._configure_environment(env.unwrapped)
            assert self._environment is env.unwrapped, "Recreate the policy when changing environments"
            episode_index = int(env.unwrapped.get_episode_index(0))
            if self._episode_index is None:
                self._exchange.start_episode(
                    env_id=0,
                    episode_index=episode_index,
                    task_instruction=self.task_description,
                    action_contract={
                        "type": "droid_abs_joint_pos",
                        "joint_names": _DROID_ARM_JOINT_NAMES,
                        "joint_position_limits": self._joint_limits.tolist(),
                        "joint_units": "radians",
                        "gripper": {"open": 0, "closed": 1},
                        "action_chunk_length": self.config.action_chunk_length,
                        "action_dim": 8,
                        "step_dt": float(env.unwrapped.step_dt),
                    },
                )
                self._episode_index = episode_index
            assert self._episode_index == episode_index, "Environment episode changed without policy.reset()"
            return self._scheduler.get_action(lambda: self._fetch_action_chunk(env.unwrapped, observation))
        except (Exception, KeyboardInterrupt) as error:
            self._exchange.record_error(error)
            raise

    def _configure_environment(self, env: gym.Env) -> None:
        """Check the action semantics before accepting controller output."""
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

        from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

        assert env.num_envs == 1, "DroidSessionPolicy currently supports num_envs=1"
        assert env.action_manager.active_terms == ["arm_action", "gripper_action"]
        arm_action = env.action_manager.get_term("arm_action")
        gripper_action = env.action_manager.get_term("gripper_action")
        assert type(arm_action) is JointPositionAction, "DroidSessionPolicy requires absolute joint position actions"
        assert type(gripper_action) is BinaryJointPositionZeroToOneAction
        assert arm_action.cfg.asset_name == gripper_action.cfg.asset_name == "robot"
        assert arm_action.action_dim == 7 and gripper_action.action_dim == 1
        assert not arm_action.cfg.use_default_offset, "Joint targets must have no default-position offset"
        assert arm_action.cfg.scale == 1.0 and arm_action.cfg.offset == 0.0 and arm_action.cfg.clip is None
        assert gripper_action.cfg.clip is None, "Gripper commands must not be transformed"
        robot = env.scene["robot"]
        joint_indices, joint_names = robot.find_joints(
            arm_action.cfg.joint_names, preserve_order=arm_action.cfg.preserve_order
        )
        assert joint_names == _DROID_ARM_JOINT_NAMES, "Unexpected DROID action joint order"
        observation_joint_names = [name for name in robot.data.joint_names if name in _DROID_ARM_JOINT_NAMES]
        assert observation_joint_names == joint_names, "DROID observations and actions must use the same joint order"
        self._joint_limits = wp.to_torch(robot.data.joint_pos_limits)[0, joint_indices].detach().cpu().numpy().copy()
        assert self._joint_limits.shape == (7, 2) and np.isfinite(self._joint_limits).all()
        assert (self._joint_limits[:, 0] < self._joint_limits[:, 1]).all()
        assert math.isfinite(env.step_dt) and env.step_dt > 0
        self._scheduler = ActionChunkScheduler(
            num_envs=1,
            action_chunk_length=self.config.action_chunk_length,
            action_horizon=self.config.action_chunk_length,
            action_dim=8,
            device=env.device,
        )
        self._environment = env

    def _fetch_action_chunk(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Export OpenPI-equivalent inputs and validate the returned joint targets."""
        extracted = self._adapter.extract(observation, env_id=0)
        packed_observation = self._adapter.pack_request(extracted, self.task_description)
        response = self._exchange.request_actions(
            observation={
                "joint_position": packed_observation["observation/joint_position"].tolist(),
                "gripper_position": packed_observation["observation/gripper_position"].tolist(),
                "simulation_step": int(env.episode_length_buf[0].item()),
                "step_dt": float(env.step_dt),
            },
            images={
                "exterior_image": packed_observation["observation/exterior_image_1_left"],
                "wrist_image": packed_observation["observation/wrist_image_left"],
            },
        )
        actions = np.asarray(response["actions"])
        assert actions.shape == (
            self.config.action_chunk_length,
            8,
        ), f"Expected actions of shape ({self.config.action_chunk_length}, 8), got {actions.shape}"
        assert np.issubdtype(actions.dtype, np.number) and not np.issubdtype(
            actions.dtype, np.complexfloating
        ), "Actions must contain real numbers"
        assert all(
            type(value) is not bool for row in response["actions"] for value in row
        ), "Actions must contain real numbers, not booleans"
        assert np.isfinite(actions).all(), "Actions must be finite"
        assert (actions[:, :7] >= self._joint_limits[:, 0]).all() and (
            actions[:, :7] <= self._joint_limits[:, 1]
        ).all(), "Joint targets exceed the robot's position limits"
        assert np.isin(actions[:, 7], [0, 1]).all(), "Gripper actions must be 0 (open) or 1 (closed)"
        return torch.as_tensor(actions, dtype=torch.float32, device=env.device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard queued actions and close the previous episode's exchange."""
        if env_ids is not None:
            if env_ids.numel() == 0:
                return
            assert (env_ids == 0).all(), "DroidSessionPolicy supports only environment zero"
        if self._episode_index is not None:
            self._exchange.end_episode()
            self._episode_index = None
        if self._scheduler is not None:
            self._scheduler.reset()

    def close(self) -> None:
        """Close the exchange and release buffered actions without starting another episode."""
        if self._closed:
            return
        try:
            if self._exchange is not None:
                self._exchange.close()
        finally:
            self._closed = True
            self._scheduler = None
            self._environment = None
