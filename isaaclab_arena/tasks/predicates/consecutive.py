# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Managed consecutive predicate base."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, TerminationTermCfg


class ConsecutivePredicate(ManagerTermBase):
    """Base for predicates that must remain true for consecutive evaluations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        consecutive_steps = cfg.params["consecutive_steps"]
        assert (
            isinstance(consecutive_steps, int) and not isinstance(consecutive_steps, bool) and consecutive_steps > 0
        ), f"consecutive_steps must be a positive integer, got {consecutive_steps!r}."
        self._required_consecutive_steps = consecutive_steps
        self.consecutive_true_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        """Current consecutive true-result count for each parallel environment."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear consecutive-result counters for selected environments."""

        ids = slice(None) if env_ids is None else env_ids
        self.consecutive_true_steps[ids] = 0

    def _update_consecutive_and_get_completion_mask(
        self,
        passed: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update active environments' streaks and return which have reached the required length."""

        passed = torch.as_tensor(passed, dtype=torch.bool, device=self.device).reshape(-1)
        assert passed.shape == (
            self.num_envs,
        ), f"Predicate returned shape {tuple(passed.shape)}; expected ({self.num_envs},)."
        if active_mask is None:
            active_mask = torch.ones_like(passed)
        else:
            active_mask = torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1)
            assert active_mask.shape == (
                self.num_envs,
            ), f"Active mask has shape {tuple(active_mask.shape)}; expected ({self.num_envs},)."
        next_count = torch.clamp(self.consecutive_true_steps + 1, max=self._required_consecutive_steps)
        updated_count = torch.where(passed, next_count, torch.zeros_like(self.consecutive_true_steps))
        self.consecutive_true_steps = torch.where(active_mask, updated_count, self.consecutive_true_steps)
        return active_mask & (self.consecutive_true_steps >= self._required_consecutive_steps)
