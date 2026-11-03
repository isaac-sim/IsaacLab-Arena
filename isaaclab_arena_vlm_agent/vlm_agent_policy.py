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
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.inference.backend import InferenceBackend, InferenceRequest, build_strict_schema
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_vlm_agent.action_adapter import AgentActionAdapter, ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.image_utils import image_to_jpeg_data_url, resize_camera_geometry
from isaaclab_arena_vlm_agent.object_pose_adapter import ObjectPoseObservationAdapter
from isaaclab_arena_vlm_agent.observation_adapter import (
    ImageObservationAdapter,
    ImageStateObservationAdapter,
    StateObservationAdapter,
    VLMObservationAdapter,
)
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


class ObservationModality(str, Enum):
    """Observations included in each VLM request."""

    # Hydra accepts enum member names, so retain the lowercase YAML vocabulary.
    image = "image"
    state = "state"
    image_state = "image_state"

    def get_adapter_type(self) -> type[VLMObservationAdapter]:
        """Return the observation adapter selected by this modality."""
        return {
            self.image: ImageObservationAdapter,
            self.state: StateObservationAdapter,
            self.image_state: ImageStateObservationAdapter,
        }[self]


@dataclass
class VLMPolicyCfg(PolicyCfg):
    """Configure policy prompts, image encoding, and action-validation attempts."""

    system_prompt: str | None
    """System message override, or None to use the selected embodiment adapter's prompt."""

    system_prompt_files: list[str] = field(default_factory=list)
    """Text files appended in order to the system prompt, such as task geometry and strategy.

    Relative paths are read from the current working directory.
    """

    image_max_width_or_height: int = 384
    """Maximum image width or height sent to the endpoint, in pixels.

    For 16:9 images, 384 yields the 384x216 resolution used for all three camera views
    in Astranomy (Sections 2 and 3.4): https://research.nvidia.com/labs/srl/projects/astranomy/
    """

    max_attempts: int = 3
    """Maximum response-validation attempts per environment step."""

    max_concurrent_requests: int = 8
    """Maximum requests in flight at once when several environments need a prediction on one step.

    Set 1 to send requests one at a time; lower it if the endpoint rate-limits concurrent requests.
    """

    vlm_embodiment_adapter: str | None = None
    """Named action and prompt preset; droid supports the droid_abs_ik embodiment."""

    observation_modality: ObservationModality = ObservationModality.image_state
    """One of image, state, or image_state; ignored when an observation adapter is injected."""

    endpoint: str | None = None
    """Inference endpoint preset, or None to use ARENA_INFERENCE_ENDPOINT."""

    model: str | None = None
    """Optional model override; image observations require a vision-capable model."""

    base_url: str | None = None
    """Optional URL override; credentials come from the endpoint's environment variable."""

    object_poses: list[str] = field(default_factory=list)
    """Scene keys of objects whose exact poses in the robot root frames are added to the prompt values.

    Wraps the modality's adapter in ObjectPoseObservationAdapter; ignored when an observation adapter is injected.
    These poses are privileged state that cameras and robot sensors do not provide.
    """

    def __post_init__(self):
        if self.system_prompt is None:
            assert self.vlm_embodiment_adapter is not None, "Set policy.system_prompt or policy.vlm_embodiment_adapter"
        else:
            assert isinstance(self.system_prompt, str) and self.system_prompt.strip(), "Set policy.system_prompt"
        assert self.vlm_embodiment_adapter in (None, "droid"), "Unknown vlm_embodiment_adapter; expected 'droid'"
        for path in self.system_prompt_files:
            assert Path(path).is_file(), f"System prompt file does not exist: '{path}'"
        self.observation_modality = ObservationModality(self.observation_modality)
        assert self.image_max_width_or_height > 0
        assert isinstance(self.max_attempts, int) and self.max_attempts >= 1
        assert (
            type(self.max_concurrent_requests) is int and self.max_concurrent_requests >= 1
        ), "max_concurrent_requests must be a positive integer"


@dataclass
class VLMChunkPolicyCfg(VLMPolicyCfg):
    """Configure command chunks executed between predictions."""

    system_prompt: str | None = None
    """Optional override for the selected embodiment's chunk prompt."""

    vlm_embodiment_adapter: str | None = "droid"
    """Named action and prompt preset."""

    chunk_size: int = field(kw_only=True)
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
            inference_backend: Backend whose ownership transfers to this policy, or None to create one.
            observation_adapter: Adapter selecting images and state, or None to construct it from config.
            action_adapter: Adapter validating responses and converting commands to native actions.
        """
        super().__init__(config)
        if observation_adapter is None:
            adapter_type = ObservationModality(config.observation_modality).get_adapter_type()
            observation_adapter = adapter_type()
            if config.object_poses:
                observation_adapter = ObjectPoseObservationAdapter(observation_adapter, config.object_poses)
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
        prompt_parts = [self._system_prompt]
        prompt_parts.extend(Path(path).read_text().strip() for path in config.system_prompt_files)
        self._system_prompt = "\n\n".join(part.strip() for part in prompt_parts if part.strip())
        self.task_description = None
        # Construct the client last so invalid adapters or schemas cannot leak a connection.
        self._inference_backend = (
            inference_backend
            if inference_backend is not None
            else InferenceBackend(
                endpoint=config.endpoint,
                model=config.model,
                base_url=config.base_url,
            )
        )
        self._request_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_requests, thread_name_prefix="vlm_inference"
        )

    def close(self) -> None:
        """Close the inference client and request threads owned by this policy."""
        # Requests still in flight after a failed step are never read, so do not wait for them.
        self._request_executor.shutdown(wait=False, cancel_futures=True)
        self._inference_backend.client.close()

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return a batch of native actions using the policy execution strategy.

        Environments that need a new prediction on this step request it concurrently, up to
        max_concurrent_requests at a time; the step returns once every request is validated.

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
        contents = {}
        for env_id in range(env.unwrapped.num_envs):
            if self._update_prediction_status(env, env_id):
                contents[env_id] = self._request_content(env, observation, env_id)
        predictions = self._infer(contents, env.unwrapped.arena_world)
        actions = []
        for env_id in range(env.unwrapped.num_envs):
            command = self._get_command(env, env_id, predictions.get(env_id))
            action = self.action_adapter.command_to_action(env, command)
            assert (
                action.ndim == 1 and action.numel() > 0 and torch.isfinite(action).all()
            ), "Expected a finite action vector"
            actions.append(action.to(device=env.unwrapped.device))
        return torch.stack(actions)

    @abstractmethod
    def _update_prediction_status(self, env: gym.Env, env_id: int) -> bool:
        """Update env_id's prediction state for this step and return whether it needs a new prediction.

        Called exactly once per environment step, before any request is sent. A strategy can record
        here how its previous prediction ended, so the next request can report it.
        """

    @abstractmethod
    def _get_command(self, env: gym.Env, env_id: int, prediction: np.ndarray | None) -> np.ndarray:
        """Return the next command, starting from prediction when one was just requested for env_id."""

    def _request_content(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> list[dict[str, Any]]:
        """Encode selected images and assemble the task and observation values into chat content."""
        images, prompt_values = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        # Injected adapters may reuse calibration dictionaries across predictions.
        prompt_values = deepcopy(prompt_values)
        encoded_images = {}
        for camera_name, image in images.items():
            url, size = image_to_jpeg_data_url(image, self.config.image_max_width_or_height)
            encoded_images[camera_name] = url
            if camera_name in prompt_values.get("camera_geometry", {}):
                resize_camera_geometry(
                    prompt_values["camera_geometry"][camera_name], (image.shape[1], image.shape[0]), size
                )
        content = [{"type": "text", "text": json.dumps({"task": self.task_description})}]
        if prompt_values:
            content.append({"type": "text", "text": json.dumps(prompt_values, allow_nan=False)})
        for camera_name, url in encoded_images.items():
            content.extend([
                {"type": "text", "text": f"camera={camera_name}"},
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                },
            ])
        return content

    def _infer(self, contents: dict[int, list[dict]], arena_world: ArenaWorld) -> dict[int, np.ndarray]:
        """Request a command per environment, allowing the model to correct locally rejected responses.

        Each attempt sends every unresolved environment's request concurrently and then validates the
        responses on the calling thread, so adapters never read the world from a worker thread.
        Each environment keeps its own conversation, so corrections never leak across environments.

        Args:
            contents: User-message content keyed by environment index.
            arena_world: Live world state passed to the action adapter's decoder.

        Returns:
            Validated commands keyed by environment index.
        """
        conversations = {}
        for env_id, content in contents.items():
            conversations[env_id] = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ]
        commands = {}
        rejections: dict[int, InvalidAgentResponseError] = {}
        attempts = 0
        while conversations and attempts < self.config.max_attempts:
            attempts += 1
            model_outputs = self._get_model_outputs(conversations)
            for env_id, model_output in model_outputs.items():
                try:
                    commands[env_id] = self._model_output_to_command(model_output, arena_world, env_id)
                except InvalidAgentResponseError as exc:
                    rejections[env_id] = exc
                    conversations[env_id].append({"role": "assistant", "content": json.dumps(model_output)})
                    conversations[env_id].append({
                        "role": "user",
                        "content": f"Invalid actions: {exc}. Return a corrected prediction.",
                    })
                else:
                    del conversations[env_id]
        if conversations:
            raise rejections[min(conversations)]
        return commands

    def _get_model_outputs(self, conversations: dict[int, list[dict]]) -> dict[int, dict[str, Any]]:
        """Send one structured-output request per conversation, up to max_concurrent_requests at a time.

        Args:
            conversations: Chat messages keyed by environment index.

        Returns:
            Parsed model outputs keyed by environment index.
        """
        requests = []
        for env_id, messages in conversations.items():
            requests.append(
                InferenceRequest(
                    messages=messages,
                    response_schema=self._response_schema,
                    schema_name="agent_actions",
                    retry_label=f"policy_action env {env_id}",
                )
            )
        # map yields results in submission order and cancels requests not yet started if one fails.
        return dict(zip(conversations, self._request_executor.map(self._inference_backend.infer, requests)))

    def _model_output_to_command(
        self, model_output: dict[str, Any], arena_world: ArenaWorld, env_id: int
    ) -> np.ndarray:
        """Validate a model output and decode it into a command for env_id.

        Args:
            model_output: Parsed JSON output returned by the endpoint.
            arena_world: Live world state passed to the action adapter's decoder.
            env_id: Environment index within the batched world state.

        Returns:
            Command with the adapter's required shape.
        """
        try:
            response = self.action_adapter.response_model.model_validate(model_output)
        except ValidationError as exc:
            raise InvalidAgentResponseError(str(exc)) from exc
        command = np.asarray(self.action_adapter.decode_command(response, arena_world, env_id), dtype=np.float32)
        # Only InvalidAgentResponseError asks the model for a correction; adapter bugs propagate unchanged.
        if not self.action_adapter.is_command_valid(command):
            raise InvalidAgentResponseError("Expected finite commands with the adapter's required shape")
        return command


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

    def _update_prediction_status(self, env: gym.Env, env_id: int) -> bool:
        return not self._chunks.get(env_id)

    def _get_command(self, env: gym.Env, env_id: int, prediction: np.ndarray | None) -> np.ndarray:
        if prediction is not None:
            self._chunks[env_id] = deque(prediction)
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
    """Use a motion generator to execute decoded goals until completion, a stall, or a step limit.

    Each request after the first reports how the previous goal ended.
    """

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
        self._previous_goals: dict[int, dict[str, Any]] = {}

    def _update_prediction_status(self, env: gym.Env, env_id: int) -> bool:
        """Record how a finished goal ended, and return whether env_id needs a new goal."""
        pending = self._goals.get(env_id)
        if pending is not None:
            outcome = self._get_goal_outcome(env, env_id, *pending)
            if outcome is None:
                return False
            self._previous_goals[env_id] = {
                "values": [round(value, 4) for value in pending[0].tolist()],
                "outcome": outcome,
                "executed_steps": pending[1],
            }
        return True

    def _get_command(self, env: gym.Env, env_id: int, prediction: np.ndarray | None) -> np.ndarray:
        if prediction is not None:
            self._goals[env_id] = (prediction, 0)
        goal, executed = self._goals[env_id]
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

    def _get_goal_outcome(self, env: gym.Env, env_id: int, goal: np.ndarray, executed: int) -> str | None:
        """Return how the pending goal ended, or None while it is still executing."""
        motion_generator = self.action_adapter.motion_generator
        if motion_generator.goal_reached(goal, env, env_id):
            return "reached"
        if motion_generator.goal_blocked(goal, env, env_id):
            return "blocked"
        if executed >= self.config.max_steps_per_goal:
            return "timeout"
        return None

    def _request_content(self, env: gym.Env, observation: dict[str, Any], env_id: int) -> list[dict[str, Any]]:
        """Report the previous goal and its outcome alongside the task description."""
        content = super()._request_content(env, observation, env_id)
        previous_goal = self._previous_goals.get(env_id)
        if previous_goal is not None:
            content[0] = {
                "type": "text",
                "text": json.dumps({"task": self.task_description, "previous_goal": previous_goal}),
            }
        return content

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending and previous goals for selected environments, or all environments."""
        if env_ids is None:
            self._goals.clear()
            self._previous_goals.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._goals.pop(env_id, None)
                self._previous_goals.pop(env_id, None)


def _get_vlm_embodiment_adapter(name: str | None):
    """Return the named action and prompt preset."""
    assert name == "droid", "Set vlm_embodiment_adapter='droid' or inject an action adapter and system_prompt"
    from isaaclab_arena_vlm_agent.droid_adapter import DroidVLMAdapter

    return DroidVLMAdapter()
