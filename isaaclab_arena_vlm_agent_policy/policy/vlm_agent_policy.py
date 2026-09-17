# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable hosted VLM agent with pluggable observation and action adapters."""

from __future__ import annotations

import base64
import importlib
import io
import json
import numpy as np
import os
import time
import torch
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from isaaclab_arena.agentic_environment_generation.inference_backend import INTERNAL_ENDPOINT
from isaaclab_arena.policy.policy_base import PolicyCfg


def encode_image(image: np.ndarray, max_edge: int = 384) -> str:
    """Return a resized RGB frame as a JPEG data URL.

    Args:
        image: HWC uint8 RGB or RGBA pixels.
        max_edge: Maximum encoded image edge in pixels.

    Returns:
        A data URL suitable for a chat completion image part.
    """
    assert image.ndim == 3 and image.shape[-1] in (
        3,
        4,
    ), f"Unexpected image shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 pixels, got {image.dtype}"
    frame = Image.fromarray(image[..., :3])
    frame.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class VLMObservationAdapter(ABC):
    """Define the camera names and proprioception exposed to a VLM agent."""

    camera_keys: tuple[str, ...]
    """Camera observation keys, in the order sent to the model."""

    def __init__(self, image_max_edge: int = 384):
        self.image_max_edge = image_max_edge

    def extract_tracking_state(self, env, observation) -> list[dict]:
        """Return per-environment EEF poses under ``eef_pose_root_xyz_xyzw`` for tracking.

        Adapters may override this to skip calibration work between decisions.
        """
        return self.extract_proprioception(env, observation)

    @abstractmethod
    def extract_proprioception(self, env, observation) -> list[dict]:
        """Return one JSON-serializable proprioception dictionary per environment."""


class AgentActionAdapter(ABC):
    """Adapt the model response contract to simulator actions."""

    response_schema: dict
    """JSON schema for the model output accepted by this adapter."""

    @abstractmethod
    def validate_environment(self, env) -> None:
        """Check that the simulator action configuration matches this adapter."""

    @abstractmethod
    def decode_command(self, payload: dict, proprioception: dict) -> np.ndarray:
        """Validate a model prediction and return one decoded command vector."""

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert one decoded command to the simulator's device and dtype."""
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)

    def tracking_error(self, previous_command: np.ndarray, proprioception: dict) -> str | None:
        """Return feedback requesting a chunk replan, or None to continue execution."""
        pass

    def compute_next_reference_pose(self, reference_pose: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
        """Compute one commanded EEF pose toward the goal using the robot's motion rules.

        Adapters used by the goal policy must implement this; chunk policies do not call it.

        Args:
            reference_pose: Previous commanded XYZ/XYZW pose in the robot root frame.
            goal_pose: Desired XYZ/XYZW pose in the same frame.

        Returns:
            The next XYZ/XYZW reference pose to send to the action adapter.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support goal reference stepping")


def _adapter_type(class_path: str, base_type: type) -> type:
    """Resolve an experiment's adapter class and check its interface."""
    assert class_path and "." in class_path, f"Configure a dotted class path for {base_type.__name__}"
    module_name, class_name = class_path.rsplit(".", 1)
    adapter_type = getattr(importlib.import_module(module_name), class_name)
    assert isinstance(adapter_type, type) and issubclass(
        adapter_type, base_type
    ), f"{class_path} must inherit {base_type.__name__}"
    return adapter_type


@dataclass
class VLMAgentPolicyCfg(PolicyCfg):
    """Configure hosted inference, image encoding, and textual decision history."""

    system_prompt: str = ""
    """Required experiment-defined system message, sent verbatim to the model."""

    observation_adapter: str = ""
    """Observation adapter class path, constructed with image_max_edge unless supplied at runtime."""

    action_adapter: str = ""
    """Action adapter class path, constructed with action_adapter_kwargs unless supplied at runtime."""

    action_adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    """Constructor arguments for the configured action adapter; runtime instances keep their own settings."""

    model: str = "openai/openai/gpt-6-astra"
    """Model served by the configured chat-completion endpoint."""

    base_url: str = INTERNAL_ENDPOINT.base_url
    """OpenAI-compatible endpoint URL."""

    api_key_env_var: str = INTERNAL_ENDPOINT.api_key_env_var
    """Environment variable containing the endpoint credential."""

    image_max_edge: int = 384
    """Maximum image edge sent to the endpoint."""

    timeout_s: float = 180.0
    """Timeout for an inference request."""

    trace_directory: str = "outputs/vlm_agent_traces"
    """Directory for a unique JSONL trace of requests and accepted commands."""

    trace_actions: bool = False
    """Record each executed command, native action, and preceding measured state."""

    decision_history: int = 0
    """Previous observation-text and accepted-response pairs retained per environment, without images."""

    max_attempts: int = 3
    """Total decode attempts for one decision, including the first request."""

    def __post_init__(self):
        assert (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ), "Set policy.system_prompt in the experiment"
        assert self.image_max_edge > 0 and self.timeout_s > 0
        assert self.decision_history >= 0
        assert self.max_attempts >= 1, f"max_attempts must be >= 1, got {self.max_attempts}"


class VLMAgentPolicy(ABC):
    """Base class to share VLM inference, correction requests, and history across action policies."""

    def __init__(
        self,
        config: VLMAgentPolicyCfg,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter | None = None,
    ):
        super().__init__(config)
        if observation_adapter is None:
            observation_adapter = _adapter_type(config.observation_adapter, VLMObservationAdapter)(
                image_max_edge=config.image_max_edge
            )
        if action_adapter is None:
            action_adapter = _adapter_type(config.action_adapter, AgentActionAdapter)(**config.action_adapter_kwargs)
        assert isinstance(observation_adapter, VLMObservationAdapter)
        assert isinstance(action_adapter, AgentActionAdapter)
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        api_key = os.environ.get(config.api_key_env_var)
        assert api_key, f"Set {config.api_key_env_var} for hosted inference"
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout_s,
            max_retries=2,
        )
        self._histories = []
        self._steps = []
        self._decisions = {}
        self.task_description = None
        self._open_trace(config.trace_directory)
        print(
            f"[vlm_agent] model={config.model} action_adapter={type(action_adapter).__name__} trace={self._trace_path}",
            flush=True,
        )

    def _open_trace(self, trace_directory: str) -> None:
        """Create a unique JSONL trace file under ``trace_directory``."""
        trace_dir = Path(trace_directory)
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = trace_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._trace = self._trace_path.open("x", encoding="utf-8", buffering=1)

    @property
    def is_remote(self) -> bool:
        return True

    @abstractmethod
    def get_action(self, env, observation) -> torch.Tensor:
        """Advance the concrete policy and request a new decision when needed."""

    def _infer(self, env_id, state):
        content = [{
            "type": "text",
            "text": json.dumps({"task": self.task_description, "proprioception": state}),
        }]
        for step, frames in self._histories[env_id]:
            for camera, url in frames.items():
                content.extend([
                    {"type": "text", "text": f"step={step}, camera={camera}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ])
        messages = [
            {"role": "system", "content": self.config.system_prompt},
        ]
        history = []
        if self.config.decision_history:
            history = self._decisions.setdefault(env_id, deque(maxlen=self.config.decision_history))
            for previous_state, previous_response in history:
                messages.extend([
                    {"role": "user", "content": previous_state},
                    {"role": "assistant", "content": previous_response},
                ])
        messages.append({"role": "user", "content": content})
        for attempt in range(self.config.max_attempts):
            start = time.monotonic()
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_completion_tokens=8192,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_actions",
                        "strict": True,
                        "schema": self.action_adapter.response_schema,
                    },
                },
            )
            choice = response.choices[0]
            raw = choice.message.content
            record = {
                "model": self.config.model,
                "action_adapter": type(self.action_adapter).__name__,
                "env_id": env_id,
                "step": self._steps[env_id],
                "history_steps": [step for step, _ in self._histories[env_id]],
                "history_decisions": len(history),
                "camera_keys": self.observation_adapter.camera_keys,
                "proprioception": state,
                "latency_s": time.monotonic() - start,
                "finish_reason": choice.finish_reason,
                "response": raw,
            }
            try:
                assert choice.finish_reason == "stop", f"Incomplete response: {choice.finish_reason}"
                assert raw, f"No actions returned; refusal={choice.message.refusal}"
                command = np.asarray(
                    self.action_adapter.decode_command(json.loads(raw), state),
                    dtype=np.float32,
                )
                assert command.ndim == 1 and command.size > 0, "Expected one command vector"
                assert np.isfinite(command).all(), "Decoded command must be finite"
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                record["validation_error"] = str(exc)
                self._trace.write(json.dumps(record) + "\n")
                if attempt == self.config.max_attempts - 1:
                    raise
                messages.append({"role": "assistant", "content": raw or ""})
                messages.append({
                    "role": "user",
                    "content": f"Invalid actions: {exc}. Return a corrected prediction.",
                })
                continue
            record["prediction"] = command.tolist()
            if self.config.decision_history:
                history.append((content[0]["text"], raw))
            self._trace.write(json.dumps(record) + "\n")
            print(
                f"[vlm_agent] env={env_id} step={self._steps[env_id]} "
                f"frames={len(self._histories[env_id])} latency={record['latency_s']:.1f}s",
                flush=True,
            )
            return command

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not self._histories:
            return
        ids = range(len(self._histories)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            self._histories[env_id].clear()
            self._steps[env_id] = 0
            getattr(self, "_decisions", {}).pop(env_id, None)

    def close(self) -> None:
        self._client.close()
        self._trace.close()
