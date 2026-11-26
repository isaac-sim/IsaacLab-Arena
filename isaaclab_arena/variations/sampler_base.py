# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from abc import ABC
from collections.abc import Callable
from typing import Any

from isaaclab.utils.configclass import configclass


@configclass
class SamplerBaseCfg:
    """Marker configclass for any sampler cfg.

    Concrete subclasses override :meth:`build` to return their live sampler.
    """

    def build(self) -> SamplerBase:
        """Return the live :class:`SamplerBase` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerBaseCfg "
            "subclass must provide a build() that returns its live sampler."
        )


class SamplerBase(ABC):
    """Base class shared by every sampler family."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[Any, torch.Tensor | None], None]] = []
        self._replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None = None
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Set the generator for random draws; ``None`` draws from torch's global random state."""
        self._generator = generator

    def set_replay_sampler(
        self,
        replay_sampler: Callable[[int, torch.Tensor | None], list[Any] | None] | None,
    ) -> None:
        """Set an optional callable that supplies recorded sample rows.

        The replay sampler receives the requested sample count and environment ids.
        Returning ``None`` keeps normal random sampling; returning rows replaces
        the random draw while preserving the sampler's public output type.
        """
        self._replay_sampler = replay_sampler

    def _get_replay_samples(self, num_samples: int, env_ids: torch.Tensor | None) -> list[Any] | None:
        """Return recorded rows for this draw, or ``None`` for live sampling."""
        if self._replay_sampler is None:
            return None
        rows = self._replay_sampler(num_samples, env_ids)
        if rows is not None:
            assert (
                len(rows) == num_samples
            ), f"Replay sampler returned {len(rows)} rows for a {num_samples}-sample draw."
        return rows

    def add_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Register ``listener``, called as ``listener(sample, env_ids)`` for every sample drawn."""
        self._listeners.append(listener)

    def _notify(self, sample: Any, env_ids: torch.Tensor | None = None) -> None:
        """Forward ``sample`` (and the ``env_ids`` it was drawn for) to every registered listener.

        ``env_ids`` is the per-env id tensor/sequence the sample's rows correspond to, or ``None``
        when the single sample applies to all envs (e.g. a build-time draw).
        """
        for listener in self._listeners:
            listener(sample, env_ids)


def sample_float_rows(
    sampler: SamplerBase,
    num_samples: int,
    shape_per_sample: tuple[int, ...],
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw ``num_samples`` numeric samples from ``sampler`` as a float32 tensor.

    ``sampler`` returns a tensor, as a UniformSampler does, or a list of number lists, as a ChoiceSampler with numeric
    values does.

    Args:
        sampler: Sampler whose ``sample(num_samples=..., env_ids=...)`` returns numeric samples.
        num_samples: Number of samples to draw.
        shape_per_sample: Shape each sample must have, for example ``(1,)`` for one number.
        env_ids: The env ids the samples correspond to, or ``None`` when the draw applies to all envs.

    Returns:
        A float32 tensor of shape ``(num_samples, *shape_per_sample)``.
    """
    samples = sampler.sample(num_samples=num_samples, env_ids=env_ids)
    if num_samples == 0:
        return torch.empty((0, *shape_per_sample), dtype=torch.float32)
    sample_rows = torch.as_tensor(samples, dtype=torch.float32)
    assert tuple(sample_rows.shape) == (num_samples, *shape_per_sample), (
        f"{type(sampler).__name__} drew samples of shape {tuple(sample_rows.shape[1:])}; this variation needs samples"
        f" of shape {tuple(shape_per_sample)}. Each value of a choice sampler is a list of"
        f" {math.prod(shape_per_sample)} numbers, for example [[0.17], [0.24]] for one number per sample."
    )
    return sample_rows
