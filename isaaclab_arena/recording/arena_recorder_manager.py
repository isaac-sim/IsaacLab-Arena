# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

import warp as wp
from isaaclab.managers.recorder_manager import RecorderManager


class ArenaRecorderManager(RecorderManager):
    """Record only assigned episodes and the environments Arena actually resets."""

    def add_to_episodes(
        self,
        key: str | None,
        value: torch.Tensor | wp.array | dict | None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Append rows belonging to active episodes, preserving their incoming order.

        Args:
            key: Dataset key, or None when a recorder term has no data.
            value: Nested tensors whose leading dimension follows env_ids.
            env_ids: Corresponding environment IDs, or None for all environments.
        """
        if not self.active_terms or key is None:
            return
        if isinstance(value, dict):
            # RecorderManager recurses through self.add_to_episodes, so filter only tensor leaves.
            super().add_to_episodes(key, value, env_ids)
            return

        requested_env_ids = self._normalize_env_ids(env_ids)
        active_positions = self._env.active_episode_mask[requested_env_ids].nonzero(as_tuple=False).flatten()
        if active_positions.numel() == 0:
            return
        if isinstance(value, wp.array):
            value = wp.to_torch(value)
        super().add_to_episodes(
            key,
            value[active_positions.to(device=value.device)],
            requested_env_ids[active_positions],
        )

    def record_pre_reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        force_export_or_skip: bool | None = None,
    ) -> None:
        """Finalize assigned episodes before Arena releases their assignments.

        Args:
            env_ids: Environments requesting a reset, or None for all environments.
            force_export_or_skip: Override the configured export behavior when provided.
        """
        if not self.active_terms:
            return
        requested_env_ids = self._normalize_env_ids(env_ids)
        finishing_env_ids = requested_env_ids[self._env.active_episode_mask[requested_env_ids]]
        if finishing_env_ids.numel() > 0:
            super().record_pre_reset(finishing_env_ids, force_export_or_skip)

    def record_post_reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Record initial states only for environments Arena actually reset.

        Args:
            env_ids: Environments requesting a reset, or None for all environments.
        """
        if not self.active_terms:
            return
        requested_env_ids = self._normalize_env_ids(env_ids)
        reset_mask = torch.isin(requested_env_ids, self._env.reset_env_ids)
        reset_mask &= self._env.active_episode_mask[requested_env_ids]
        reset_env_ids = requested_env_ids[reset_mask]
        if reset_env_ids.numel() > 0:
            super().record_post_reset(reset_env_ids)

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Return environment IDs on the environment's device."""
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self._env.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)
