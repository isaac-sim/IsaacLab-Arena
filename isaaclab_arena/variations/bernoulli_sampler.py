# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class BernoulliSamplerCfg(SamplerBaseCfg):
    """Config for :class:`BernoulliSampler`."""

    probability: float = 0.5
    """Probability that a draw returns ``True``."""

    def build(self) -> BernoulliSampler:
        return BernoulliSampler(probability=self.probability)


class BernoulliSampler(SamplerBase):
    """Sampler returning independent ``True``/``False`` draws at a fixed probability."""

    def __init__(self, probability: float):
        super().__init__()
        assert 0.0 <= probability <= 1.0, f"probability must be in [0, 1]; got {probability}."
        self.probability = probability

    def sample(self, num_samples: int, env_ids: torch.Tensor | None = None) -> list[bool]:
        """Draw ``num_samples`` independent booleans, ``True`` with probability ``self.probability``.

        Args:
            num_samples: Number of independent samples to draw.
            env_ids: The env ids the drawn values correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A ``list`` of length ``num_samples`` of booleans.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        result = (torch.rand(num_samples) < self.probability).tolist()
        self._notify(result, env_ids)
        return result
