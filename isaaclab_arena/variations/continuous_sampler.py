# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import abstractmethod

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class ContinuousSamplerCfg(SamplerBaseCfg):
    """Config for ContinuousSampler."""

    def build(self) -> ContinuousSampler:
        """Return the live ContinuousSampler described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete ContinuousSamplerCfg "
            "subclass must provide a build() that returns its live ContinuousSampler."
        )


class ContinuousSampler(SamplerBase):
    """Draws continuous numeric values from a fixed-shape distribution.

    Concrete subclasses implement ``_sample`` to return a tensor of shape
    ``(num_samples, *shape_per_sample)``.
    """

    def sample(self, num_samples: int, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw, typically the
                number of environments we're drawing a sample for.
            env_ids: The env ids the drawn rows correspond to, forwarded to sample listeners so
                they can attribute values per env. ``None`` when the draw applies to all envs.

        Returns:
            A tensor of shape ``(num_samples, *shape_per_sample)``.
        """
        result = self._sample(num_samples)
        self._notify(result, env_ids)
        return result

    @abstractmethod
    def _sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` values as a tensor of shape ``(num_samples, *shape_per_sample)``."""
        ...

    @property
    @abstractmethod
    def shape_per_sample(self) -> torch.Size:
        """Shape of a single sample."""
        ...
