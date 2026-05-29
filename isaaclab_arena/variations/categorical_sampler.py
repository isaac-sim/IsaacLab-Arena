# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Uniform categorical sampler over an arbitrary ``choices`` sequence.

The choice domain is not a property of the distribution: it's a per-call
argument passed to :meth:`SamplerBase.sample` (as ``choices``), since
callers typically know the pool only at apply time (e.g. the HDRs to pick
from). The returned samples are the actual items from ``choices``, not
indices, so downstream consumers (e.g. the variation recorder) capture
meaningful values directly.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Any

from isaaclab.utils import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class CategoricalSamplerCfg(SamplerBaseCfg):
    """Config for :class:`CategoricalSampler`.

    Intentionally empty: the categorical distribution has no declarative
    parameters in this implementation. ``choices`` is supplied per call via
    :meth:`SamplerBase.sample`.
    """

    def build(self) -> CategoricalSampler:
        return CategoricalSampler()


class CategoricalSampler(SamplerBase):
    """Uniform categorical sampler returning items drawn from a per-call ``choices`` sequence.

    Each call to :meth:`sample` returns a ``list`` of length ``num_samples``,
    where each item is uniformly drawn (with replacement) from ``choices``.
    """

    def _sample(self, num_samples: int, choices: Sequence[Any], **kwargs) -> list[Any]:  # noqa: ARG002
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "CategoricalSampler requires a non-empty 'choices' sequence."
        indices = torch.randint(low=0, high=len(choices), size=(num_samples,), dtype=torch.long)
        return [choices[int(i)] for i in indices]
