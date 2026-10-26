# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hosted VLM inference with configurable observation and action adapters."""

from __future__ import annotations

import gymnasium as gym
import json
import numpy as np
import torch
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from isaaclab_arena.inference.backend import InferenceBackend, InferenceRequest, build_strict_schema
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url
from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


@dataclass
class VLMPolicyCfg(PolicyCfg):
    """Configure policy prompts, image encoding, and action-validation attempts."""

    system_prompt: str
    """Required system message defining the task, units, and response contract."""

    image_max_width_or_height: int = 384
    """Maximum image width or height sent to the endpoint, in pixels.

    For 16:9 images, 384 yields the 384x216 resolution used for all three camera views
    in Astranomy (Sections 2 and 3.4): https://research.nvidia.com/labs/srl/projects/astranomy/
    """

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    def __post_init__(self):
        assert (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ), "Set policy.system_prompt in the experiment"
        assert self.image_max_width_or_height > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1


@dataclass
class VLMGoalPolicyCfg(VLMPolicyCfg):
    """Configure VLM goal prediction and the execution budget for each goal."""

    max_steps_per_goal: int = field(kw_only=True)
    """Maximum commands to execute before requesting a new goal."""

    def __post_init__(self):
        super().__post_init__()
        assert (
            type(self.max_steps_per_goal) is int and self.max_steps_per_goal > 0
        ), "max_steps_per_goal must be a positive integer"


class VLMPolicy(PolicyBase[VLMPolicyCfg]):
    """Share VLM request preparation and response validation across execution strategies.

    Each request contains the observations selected by the adapter and the task description.
    """

    def __init__(
        self,
        config: VLMPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: AgentActionAdapter,
    ):
        """Create a policy with a shared inference backend and stateless adapters.

        Args:
            config: Prompt, image encoding, and action-validation settings.
            inference_backend: Caller-owned backend shared with other inference steps.
                The caller is responsible for closing its client.
            observation_adapter: Adapter selecting camera images and prompt values for the VLM.
            action_adapter: Adapter validating responses and converting commands to native actions.
        """
        super().__init__(config)
        assert isinstance(observation_adapter, VLMObservationAdapter)
        assert isinstance(action_adapter, AgentActionAdapter)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self._response_schema = build_strict_schema(action_adapter.response_model)
        self._inference_backend = inference_backend
        self.task_description = None

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return a batch of native actions using the policy execution strategy.

        Args:
            env: Arena environment, optionally gym-wrapped.
            observation: Batched observations consumed by the observation adapter.

        Returns:
            Actions shaped (num_envs, action_dim) on the environment device.
        """
        assert self.task_description, "Set the task description before requesting actions"
        assert self.action_adapter.is_robot_action_space_supported(
            env
        ), "Robot action space is unsupported by this adapter"
        actions = []
        for env_id in range(env.unwrapped.num_envs):
            command = self._get_command(env, observation, env_id)
            action = self.action_adapter.command_to_action(env, command)
            assert (
                action.ndim == 1 and action.numel() > 0 and torch.isfinite(action).all()
            ), "Expected a finite action vector"
            actions.append(action.to(device=env.unwrapped.device))
        return torch.stack(actions)

    @abstractmethod
    def _get_command(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> np.ndarray:
        """Return one command using the current observation and any pending prediction."""

    def _request_content(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> list[dict[str, Any]]:
        """Encode selected images and assemble the task and observation values into chat content."""
        images, prompt_values = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        content = [{"type": "text", "text": json.dumps({"task": self.task_description})}]
        if prompt_values:
            content.append({"type": "text", "text": json.dumps(prompt_values, allow_nan=False)})
        for camera_name, image in images.items():
            content.extend([
                {"type": "text", "text": f"camera={camera_name}"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_jpeg_data_url(image, self.config.image_max_width_or_height)},
                },
            ])
        return content

    def _infer(self, content: list[dict], arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Request a command, allowing the model to correct locally rejected responses."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content},
        ]
        rejection: InvalidAgentResponseError | None = None
        for attempt in range(self.config.max_attempts):
            payload = self._inference_backend.infer(
                InferenceRequest(
                    messages=messages,
                    response_schema=self._response_schema,
                    schema_name="agent_actions",
                    retry_label="policy_action",
                )
            )
            try:
                response = self.action_adapter.response_model.model_validate(payload)
            except ValidationError as exc:
                rejection = InvalidAgentResponseError(str(exc))
            else:
                try:
                    command = np.asarray(
                        self.action_adapter.decode_command(response, arena_world, env_id), dtype=np.float32
                    )
                except InvalidAgentResponseError as exc:
                    rejection = exc
                else:
                    # Adapter implementation errors must propagate without asking the model to fix them.
                    if self.action_adapter.is_command_valid(command):
                        return command
                    rejection = InvalidAgentResponseError("Expected finite commands with the adapter's required shape")
            messages.append({"role": "assistant", "content": json.dumps(payload)})
            messages.append({
                "role": "user",
                "content": f"Invalid actions: {rejection}. Return a corrected prediction.",
            })
        assert rejection is not None, "Expected a rejection after exhausting response-validation attempts"
        raise rejection


class VLMStepPolicy(VLMPolicy):
    """Predict one action per environment from fresh observations on every step."""

    def __init__(
        self,
        config: VLMPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: AgentActionAdapter,
    ):
        assert not isinstance(action_adapter, (ChunkActionAdapter, GoalActionAdapter)), "Expected a single-step adapter"
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )

    def _get_command(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> np.ndarray:
        return self._infer(self._request_content(env, observation, env_id), env.unwrapped.arena_world, env_id)


class VLMChunkPolicy(VLMPolicy):
    """Execute a validated command chunk before requesting another from current observations."""

    def __init__(
        self,
        config: VLMPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: ChunkActionAdapter,
    ):
        assert isinstance(action_adapter, ChunkActionAdapter)
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )
        self._chunks: dict[int, deque] = {}

    def _get_command(self, env, observation, env_id: int) -> np.ndarray:
        if not self._chunks.get(env_id):
            chunk = self._infer(self._request_content(env, observation, env_id), env.unwrapped.arena_world, env_id)
            self._chunks[env_id] = deque(chunk)
        return self._chunks[env_id].popleft()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending commands for selected environments, or all environments."""
        if env_ids is None:
            self._chunks.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._chunks.pop(env_id, None)


class VLMGoalPolicy(VLMPolicy, PolicyBase[VLMGoalPolicyCfg]):
    """Use a motion generator to execute decoded goals until completion or a step limit."""

    def __init__(
        self,
        config: VLMGoalPolicyCfg,
        *,
        inference_backend: InferenceBackend,
        observation_adapter: VLMObservationAdapter,
        action_adapter: GoalActionAdapter,
    ):
        assert isinstance(action_adapter, GoalActionAdapter)
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )
        self._goals: dict[int, tuple[np.ndarray, int]] = {}

    def _get_command(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> np.ndarray:
        pending = self._goals.get(env_id)
        if (
            pending is None
            or pending[1] >= self.config.max_steps_per_goal
            or self.action_adapter.motion_generator.goal_reached(pending[0], env, env_id)
        ):
            goal = self._infer(self._request_content(env, observation, env_id), env.unwrapped.arena_world, env_id)
            executed = 0
        else:
            goal, executed = pending
        command = np.asarray(
            self.action_adapter.motion_generator.compute_command(goal, env, env_id),
            dtype=np.float32,
        )
        expected_dim = env.unwrapped.action_dim
        assert (
            command.shape == (expected_dim,) and np.isfinite(command).all()
        ), f"Expected a finite goal command with dimension {expected_dim}, got shape {command.shape}"
        self._goals[env_id] = (goal, executed + 1)
        return command

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending goals for selected environments, or all environments."""
        if env_ids is None:
            self._goals.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._goals.pop(env_id, None)
