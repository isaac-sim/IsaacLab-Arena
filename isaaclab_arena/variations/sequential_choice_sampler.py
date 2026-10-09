# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Generic, TypeVar

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg

T = TypeVar("T")


@configclass
class SequentialChoiceSamplerCfg(SamplerBaseCfg):
    """Configuration for cycling through choices in declaration order."""

    def build(self) -> SequentialChoiceSampler:
        return SequentialChoiceSampler()


class SequentialChoiceSampler(SamplerBase, Generic[T]):
    """Cycle through per-call choices, starting with the first choice on every call."""

    def sample(
        self,
        num_samples: int,
        choices: Sequence[T],
        env_ids: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> list[T]:
        """Return choices in repeated declaration order, or the supplied replay rows.

        Args:
            num_samples: Number of samples to return.
            choices: Non-empty sequence to cycle through.
            env_ids: Environment ids forwarded to sample listeners.
            generator: Accepted for compatibility with ChoiceSampler; sequential draws do not use it.

        Returns:
            A list containing one choice per requested sample.
        """
        assert num_samples >= 0, f"num_samples must be non-negative; got {num_samples}."
        assert len(choices) >= 1, "SequentialChoiceSampler requires a non-empty 'choices' sequence."
        replay_samples = self._get_replay_samples(num_samples, env_ids)
        if replay_samples is not None:
            assert all(value in choices for value in replay_samples), "Choice replay samples must belong to 'choices'."
            result = replay_samples
        else:
            result = [choices[index % len(choices)] for index in range(num_samples)]
        self._notify(result, env_ids)
        return result
