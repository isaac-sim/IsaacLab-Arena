# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import abstractmethod

from isaaclab.utils import configclass

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

    @property
    @abstractmethod
    def shape_per_sample(self) -> torch.Size:
        """Shape of a single sample."""
        ...
