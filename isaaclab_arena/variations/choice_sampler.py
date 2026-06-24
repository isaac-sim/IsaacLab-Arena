# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Generic, TypeVar

from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

T = TypeVar("T")


@configclass
class ChoiceSamplerCfg(SamplerBaseCfg):
    """Config for ChoiceSampler (empty; ``choices`` is supplied per call)."""

    def build(self) -> ChoiceSampler:
        return ChoiceSampler()


class ChoiceSampler(SamplerBase, Generic[T]):
    """Uniform sampler returning items drawn from a per-call ``choices`` sequence."""

    def sample(self, num_samples: int, choices: Sequence[T], env_ids: torch.Tensor | None = None) -> list[T]:
        """Draw ``num_samples`` items from ``choices``.

        Args:
            num_samples: Number of independent samples to draw, typically the
                number of environments we're drawing a sample for.
            choices: Pool of items to draw from. Must be non-empty.
            env_ids: The env ids the drawn items correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of items drawn from ``choices``.
        """
        result = self._sample(num_samples, choices)
        self._notify(result, env_ids)
        return result

    def _sample(self, num_samples: int, choices: Sequence[T]) -> list[T]:
        """Draw ``num_samples`` items from ``choices`` (non-empty)."""
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "ChoiceSampler requires a non-empty 'choices' sequence."
        indices = torch.randint(low=0, high=len(choices), size=(num_samples,))
        return [choices[int(i)] for i in indices]
