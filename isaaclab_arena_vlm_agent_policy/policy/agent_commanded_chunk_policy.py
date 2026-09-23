# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent-commanded EEF chunks with optional textual decision history."""

import json
import torch
from collections import deque
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy import (
    AgentActionAdapter,
    VLMAgentPolicy,
    VLMAgentPolicyCfg,
    VLMObservationAdapter,
    encode_image,
)


@dataclass
class AgentCommandedChunkPolicyCfg(VLMAgentPolicyCfg):
    """Configure calibrated chunks and optional retained textual decisions."""

    trace_actions: bool = True
    chunk_size: int = 15
    """Number of pose-and-gripper commands predicted per inference."""

    def __post_init__(self):
        super().__post_init__()
        assert self.chunk_size >= 1, f"chunk_size must be >= 1, got {self.chunk_size}"


@register_policy
class AgentCommandedChunkPolicy(VLMAgentPolicy, PolicyBase[AgentCommandedChunkPolicyCfg]):
    """Replay a fixed-length EEF chunk using adapters from config or constructor arguments.

    The action adapter decodes ``chunk_size`` flattened XYZ/XYZW pose-and-gripper commands
    (default 15) and converts each eight-value command to the robot's native action format.
    """

    name = "agent_commanded_chunk"

    def __init__(
        self,
        config: AgentCommandedChunkPolicyCfg,
        *,
        observation_adapter: VLMObservationAdapter | None = None,
        action_adapter: AgentActionAdapter | None = None,
    ):
        super().__init__(config, observation_adapter, action_adapter)
        if hasattr(self.action_adapter, "chunk_size"):
            self.action_adapter.chunk_size = config.chunk_size
        self._chunks = []
        self._indices = []

    def get_action(self, env, observation):
        base = env.unwrapped
        assert self.task_description
        self.action_adapter.validate_environment(env)
        if not self._histories:
            self._histories = [deque(maxlen=1) for _ in range(base.num_envs)]
            self._steps = [0] * base.num_envs
            self._chunks = [None] * base.num_envs
            self._indices = [0] * base.num_envs
        assert len(self._chunks) == base.num_envs
        states = self.observation_adapter.extract_tracking_state(env, observation)
        calibrated_states = None
        actions = []
        for env_id, state in enumerate(states):
            feedback = None
            if self._chunks[env_id] is not None and self._indices[env_id] > 0:
                previous = self._chunks[env_id][self._indices[env_id] - 1]
                feedback = self.action_adapter.tracking_error(previous, state)
                if feedback is not None:
                    self._trace.write(
                        json.dumps({
                            "event": "tracking_replan",
                            "env_id": env_id,
                            "step": self._steps[env_id],
                            "reason": feedback,
                            "discarded_actions": self.config.chunk_size - self._indices[env_id],
                        })
                        + "\n"
                    )
                    self._chunks[env_id] = None
            if self._chunks[env_id] is None or self._indices[env_id] == self.config.chunk_size:
                if calibrated_states is None:
                    calibrated_states = self.observation_adapter.extract_proprioception(env, observation)
                state = calibrated_states[env_id]
                if feedback is not None:
                    state["controller_feedback"] = feedback
                frames = {
                    key: encode_image(observation["camera_obs"][key][env_id].cpu().numpy(), self.config.image_max_edge)
                    for key in self.observation_adapter.camera_keys
                }
                self._histories[env_id].append((self._steps[env_id], frames))
                self._chunks[env_id] = self._infer(env_id, state).reshape(self.config.chunk_size, 8)
                self._indices[env_id] = 0
            command = self._chunks[env_id][self._indices[env_id]]
            action = self.action_adapter.command_to_action(env, command)
            if self.config.trace_actions:
                self._trace.write(
                    json.dumps({
                        "event": "execute",
                        "env_id": env_id,
                        "step": self._steps[env_id],
                        "proprioception": states[env_id],
                        "command": command.tolist(),
                        "native_action": action.cpu().tolist(),
                    })
                    + "\n"
                )
            actions.append(action)
            self._indices[env_id] += 1
            self._steps[env_id] += 1
        return torch.stack(actions)

    def reset(self, env_ids=None):
        if not self._histories:
            return
        ids = range(len(self._histories)) if env_ids is None else env_ids.reshape(-1).tolist()
        for env_id in ids:
            self._chunks[env_id] = None
            self._indices[env_id] = 0
        super().reset(env_ids)
