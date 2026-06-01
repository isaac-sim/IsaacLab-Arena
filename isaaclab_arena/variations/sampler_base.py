# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler abstract base for the variation system.

A :class:`SamplerBase` is a stateless description of how values are drawn,
paired with a declarative :class:`SamplerBaseCfg` that ``build()`` s into it.

:meth:`SamplerBase.sample` returns ``Any``: continuous samplers return a
``torch.Tensor`` of shape ``(num_samples, *shape_per_sample)``; categorical
samplers return a ``list``.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Any

from isaaclab.utils import configclass


@configclass
class SamplerBaseCfg:
    """Base configclass for declarative sampler descriptions."""

    def build(self) -> SamplerBase:
        """Return the live :class:`SamplerBase` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerBaseCfg "
            "subclass must provide a build() that returns its live SamplerBase."
        )


class SamplerBase(ABC):
    """Baseclass for all samplers. Stateless"""

    @property
    @abstractmethod
    def shape_per_sample(self) -> torch.Size:
        """Shape of a single sample."""
        ...

    def sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution.

        Args:
            num_samples: Number of independent samples to draw, typically the
                number of environments we're drawing a sample for.
            **kwargs: Extra per-call arguments forwarded to the concrete sampler
                (e.g. ``choices`` for a categorical sampler).

        Returns:
            Either a tensor of shape ``(num_samples, *shape_per_sample)`` for tensor
            samplers, or a ``list`` of length ``num_samples * shape_per_sample`` for categorical
            samplers.
        """
        return self._sample(num_samples, **kwargs)

    @abstractmethod
    def _sample(self, num_samples: int, **kwargs) -> Any:
        """Draw ``num_samples`` values from this distribution."""
        ...
