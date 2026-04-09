from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.client_state import ClientState
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


class EchoServerSidePolicy(ServerSidePolicy):
    """Minimal server policy for transport-first validation."""

    def __init__(self) -> None:
        super().__init__(config=None)

    def _build_protocol(self) -> ChunkingActionProtocol:
        return ChunkingActionProtocol(
            action_dim=2,
            observation_keys=["obs"],
            action_chunk_length=1,
            action_horizon=1,
        )

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> EchoServerSidePolicy:
        del args
        return EchoServerSidePolicy()

    def _resolve_task_descriptions(
        self,
        env_ids: list[int] | None,
        client_state: ClientState | None,
        *,
        batch_size: int,
    ) -> list[str | None]:
        if client_state is None:
            fallback = self._task_description
            return [fallback] * max(batch_size, 1)

        target_env_ids = env_ids if env_ids is not None else list(range(batch_size))
        if not target_env_ids:
            target_env_ids = [0]
        return [client_state.instructions[env_id] for env_id in target_env_ids]

    def get_action(
        self,
        observation: dict[str, Any],
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        obs = observation["obs"]
        batch_size = int(obs.shape[0]) if hasattr(obs, "shape") and len(obs.shape) > 0 else 1
        descriptions = self._resolve_task_descriptions(env_ids, client_state, batch_size=batch_size)

        info: dict[str, Any] = {
            "task_descriptions": descriptions,
            "obs_shape": list(obs.shape) if hasattr(obs, "shape") else None,
        }
        if isinstance(obs, torch.Tensor):
            info.update(
                {
                    "obs_backend": "torch",
                    "obs_device": str(obs.device),
                    "obs_dtype": str(obs.dtype),
                    "is_cuda": bool(obs.is_cuda),
                }
            )
        else:
            info.update(
                {
                    "obs_backend": type(obs).__name__,
                    "obs_device": "cpu",
                    "obs_dtype": str(getattr(obs, "dtype", type(obs).__name__)),
                    "is_cuda": False,
                }
            )

        action = np.zeros((batch_size, 1, 2), dtype=np.float32)
        return {"action": action}, info

    def reset(
        self,
        env_ids: list[int] | None = None,
        reset_options: dict[str, Any] | None = None,
        *,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        del reset_options, client_state
        return {"status": "reset_success", "env_ids": env_ids}
