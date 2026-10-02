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

    def __init__(self) -> None:
        super().__init__()
        self._required_range: tuple[tuple[int, ...], float | None, float | None, bool] | None = None

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
        context = self._sampling_context
        if context is None:
            result = self._sample(num_samples)
        else:
            keys = context.episode_keys(num_samples, env_ids)
            rows = []
            for key in keys:
                if context.replay is not None:
                    row = self._replay_value(context.replay.value(self._variation_path, key))
                else:
                    row = self._sample_with_generator(context.generator(self._variation_path, key))
                rows.append(row)
            result = torch.stack(rows) if rows else torch.empty((0, *self.shape_per_sample), dtype=torch.float32)
        self._validate_sample(result, num_samples)
        self._notify(result, env_ids)
        return result

    def validate_range(
        self,
        shape: tuple[int, ...],
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        """Declare the variation's required output domain without drawing a sample.

        The base checks declared shape now and validates each realized batch before
        notifying listeners. Distribution-aware samplers may override this method,
        call ``super()``, and also reject impossible configured bounds before a draw.
        """
        assert tuple(self.shape_per_sample) == shape, f"Expected sampler shape {shape}."
        self._required_range = (shape, minimum, maximum, minimum_inclusive)

    def _validate_sample(self, sample: torch.Tensor, num_samples: int) -> None:
        """Reject invalid realized values before recording or applying the batch."""
        if self._required_range is None:
            return
        shape, minimum, maximum, minimum_inclusive = self._required_range
        assert isinstance(sample, torch.Tensor) and sample.dtype != torch.bool, "Expected a numeric tensor sample."
        assert tuple(sample.shape) == (num_samples, *shape), f"Expected sample shape {(num_samples, *shape)}."
        assert torch.isfinite(sample).all(), "Sampled values must be finite."
        if minimum is not None:
            valid = sample >= minimum if minimum_inclusive else sample > minimum
            assert valid.all(), f"Sampled values violate the physical minimum {minimum}."
        if maximum is not None:
            assert (sample <= maximum).all(), f"Sampled values violate the physical maximum {maximum}."

    def _sample_with_generator(self, generator: torch.Generator) -> torch.Tensor:
        """Draw one row with the supplied generator; custom samplers opt in explicitly."""
        raise NotImplementedError(f"{type(self).__name__} does not implement contextual sampling.")

    def _replay_value(self, value) -> torch.Tensor:
        """Validate one recorded numeric row without accepting booleans or shape coercion."""

        def numeric(item):
            if isinstance(item, list):
                return all(numeric(child) for child in item)
            return type(item) in (int, float)

        assert numeric(value), f"Replay value for '{self._variation_path}' must contain only numbers."
        result = torch.tensor(value, dtype=torch.float32)
        assert result.shape == self.shape_per_sample, (
            f"Replay shape for '{self._variation_path}' must be {tuple(self.shape_per_sample)}, got"
            f" {tuple(result.shape)}."
        )
        assert torch.isfinite(result).all(), f"Replay value for '{self._variation_path}' must be finite."
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
