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

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.inference.backend import MAX_RETRIES_LIMIT, InferenceBackend, InferenceRequest, build_strict_schema
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url
from isaaclab_arena_vlm_agent.observation_adapter import (
    ImageObservationAdapter,
    ImageStateObservationAdapter,
    StateObservationAdapter,
    VLMObservationAdapter,
)
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


@dataclass
class VLMPolicyCfg(PolicyCfg):
    """Configure policy prompts, image encoding, and action-validation attempts."""

    system_prompt: str | None
    """System message override, or None to use the selected embodiment adapter's prompt."""

    image_max_width_or_height: int = 384
    """Maximum image width or height sent to the endpoint, in pixels.

    For 16:9 images, 384 yields the 384x216 resolution used for all three camera views
    in Astranomy (Sections 2 and 3.4): https://research.nvidia.com/labs/srl/projects/astranomy/
    """

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    vlm_embodiment_adapter: str | None = None
    """Named action and prompt preset; droid supports the droid_abs_ik embodiment."""

    observation_modality: str = "image_state"
    """One of image, state, or image_state; ignored when an observation adapter is injected."""

    endpoint: str | None = None
    """Inference endpoint preset, or None to use ARENA_INFERENCE_ENDPOINT."""

    model: str | None = None
    """Optional model override; image observations require a vision-capable model."""

    base_url: str | None = None
    """Optional URL override; credentials come from the endpoint's environment variable."""

    temperature: float = 0.2
    """Sampling temperature when supported by the endpoint."""

    max_tokens: int = 4096
    """Maximum output tokens per inference request."""

    max_retries: int = 3
    """Additional transport attempts, separate from response-validation max_attempts."""

    def __post_init__(self):
        if self.system_prompt is None:
            assert self.vlm_embodiment_adapter is not None, "Set policy.system_prompt or policy.vlm_embodiment_adapter"
        else:
            assert isinstance(self.system_prompt, str) and self.system_prompt.strip(), "Set policy.system_prompt"
        assert self.vlm_embodiment_adapter in (None, "droid"), "Unknown vlm_embodiment_adapter; expected 'droid'"
        assert self.observation_modality in {"image", "state", "image_state"}
        assert self.image_max_width_or_height > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1
        assert type(self.max_tokens) is int and self.max_tokens > 0
        assert type(self.max_retries) is int and 0 <= self.max_retries < MAX_RETRIES_LIMIT


@dataclass
class VLMStepPolicyCfg(VLMPolicyCfg):
    """Configure one command per prediction."""

    system_prompt: str | None = None
    """Optional override for the selected embodiment's single-command prompt."""

    vlm_embodiment_adapter: str | None = "droid"
    """Named action and prompt preset."""


@dataclass
class VLMChunkPolicyCfg(VLMPolicyCfg):
    """Configure command chunks executed between predictions."""

    system_prompt: str | None = None
    """Optional override for the selected embodiment's chunk prompt."""

    vlm_embodiment_adapter: str | None = "droid"
    """Named action and prompt preset."""

    chunk_size: int = 1
    """Number of commands per prediction; also used by the default prompt."""

    def __post_init__(self):
        super().__post_init__()
        assert type(self.chunk_size) is int and self.chunk_size > 0, "chunk_size must be a positive integer"


@dataclass
class VLMGoalPolicyCfg(VLMPolicyCfg):
    """Configure VLM goal prediction and the execution budget for each goal."""

    system_prompt: str | None = None
    """Optional override for the selected embodiment's goal prompt."""

    vlm_embodiment_adapter: str | None = "droid"
    """Named action, motion-generator, gripper, and prompt preset."""

    max_steps_per_goal: int = field(kw_only=True)
    """Maximum commands to execute before requesting a new goal."""

    position_tolerance_m: float = 0.01
    """Maximum translation error for the configured pose motion generator."""

    rotation_tolerance_rad: float = 0.1
    """Maximum rotation error for the configured pose motion generator."""

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
        inference_backend: InferenceBackend | None = None,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter,
    ):
        """Create configured dependencies, or use explicitly injected dependencies.

        Args:
            config: Prompt, image encoding, and action-validation settings.
            inference_backend: Caller-owned backend, or None to create a client owned by this policy.
            observation_adapter: Adapter selecting images and state, or None to construct it from config.
            action_adapter: Adapter validating responses and converting commands to native actions.
        """
        super().__init__(config)
        if observation_adapter is None:
            adapter_type = {
                "image": ImageObservationAdapter,
                "state": StateObservationAdapter,
                "image_state": ImageStateObservationAdapter,
            }[config.observation_modality]
            observation_adapter = adapter_type()
        assert isinstance(observation_adapter, VLMObservationAdapter)
        assert isinstance(action_adapter, AgentActionAdapter)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self._response_schema = build_strict_schema(action_adapter.response_model)
        self._system_prompt = config.system_prompt
        if self._system_prompt is None:
            self._system_prompt = _get_vlm_embodiment_adapter(config.vlm_embodiment_adapter).get_system_prompt(
                action_adapter
            )
        self.task_description = None
        self._owns_inference_backend = inference_backend is None
        self._closed = False
        # Construct the client last so invalid adapters or schemas cannot leak a connection.
        self._inference_backend = (
            inference_backend
            if inference_backend is not None
            else InferenceBackend(
                endpoint=config.endpoint,
                model=config.model,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                max_retries=config.max_retries,
            )
        )

    def close(self) -> None:
        """Discard pending commands and close only a policy-owned inference client."""
        if not self._closed:
            self.reset()
            if self._owns_inference_backend:
                self._inference_backend.client.close()
            self._closed = True

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
            {"role": "system", "content": self._system_prompt},
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


@register_policy
class VLMStepPolicy(VLMPolicy, PolicyBase[VLMStepPolicyCfg]):
    """Predict one action per environment from fresh observations on every step."""

    name = "vlm_step"

    def __init__(
        self,
        config: VLMPolicyCfg,
        *,
        inference_backend: InferenceBackend | None = None,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter | None = None,
    ):
        if action_adapter is None:
            action_adapter = _get_vlm_embodiment_adapter(config.vlm_embodiment_adapter).make_step_action_adapter()
        assert not isinstance(action_adapter, (ChunkActionAdapter, GoalActionAdapter)), "Expected a single-step adapter"
        super().__init__(
            config,
            inference_backend=inference_backend,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
        )

    def _get_command(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> np.ndarray:
        return self._infer(self._request_content(env, observation, env_id), env.unwrapped.arena_world, env_id)


@register_policy
class VLMChunkPolicy(VLMPolicy, PolicyBase[VLMChunkPolicyCfg]):
    """Execute a validated command chunk before requesting another from current observations."""

    name = "vlm_chunk"

    def __init__(
        self,
        config: VLMPolicyCfg,
        *,
        inference_backend: InferenceBackend | None = None,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: ChunkActionAdapter | None = None,
    ):
        if action_adapter is None:
            assert isinstance(config, VLMChunkPolicyCfg), "Configure VLMChunkPolicyCfg or inject an action adapter"
            action_adapter = _get_vlm_embodiment_adapter(config.vlm_embodiment_adapter).make_chunk_action_adapter(
                config.chunk_size
            )
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


@register_policy
class VLMGoalPolicy(VLMPolicy, PolicyBase[VLMGoalPolicyCfg]):
    """Use a motion generator to execute decoded goals until completion or a step limit."""

    name = "vlm_goal"

    def __init__(
        self,
        config: VLMGoalPolicyCfg,
        *,
        inference_backend: InferenceBackend | None = None,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: GoalActionAdapter | None = None,
    ):
        if action_adapter is None:
            action_adapter = _get_vlm_embodiment_adapter(config.vlm_embodiment_adapter).make_goal_action_adapter(
                config.position_tolerance_m, config.rotation_tolerance_rad
            )
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
        expected_dim = env.unwrapped.action_manager.total_action_dim
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


def _get_vlm_embodiment_adapter(name: str | None):
    """Return the named action and prompt preset."""
    assert name == "droid", "Set vlm_embodiment_adapter='droid' or inject an action adapter and system_prompt"
    from isaaclab_arena_vlm_agent.droid_adapter import DroidVLMAdapter

    return DroidVLMAdapter()
