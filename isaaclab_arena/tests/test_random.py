# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

import pytest

from isaaclab_arena.utils.random import get_random_rotation, get_rngs


def test_get_random_rotation_in_range():
    """Sampled yaws always fall within [-pi, pi)."""
    generator = torch.Generator().manual_seed(0)
    samples = [get_random_rotation(generator) for _ in range(1000)]
    assert all(-math.pi <= yaw < math.pi for yaw in samples)


def _sample_sequence(seed: int, n: int = 5) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    return [get_random_rotation(generator) for _ in range(n)]


def test_get_random_rotation_seed_determinism():
    """Equally-seeded generators produce identical sequences; different seeds differ."""
    assert _sample_sequence(42) == _sample_sequence(42)
    assert _sample_sequence(42) != _sample_sequence(7)


def _rng_streams(rngs, n: int = 5) -> list[list[float]]:
    return [[rng.random() for _ in range(n)] for rng in rngs]


def test_get_rngs_same_seed_is_reproducible():
    """A given base_seed must deterministically reproduce every per-env stream."""
    assert _rng_streams(get_rngs(4, 42)) == _rng_streams(get_rngs(4, 42))


def test_get_rngs_streams_are_independent():
    """Per-env RNGs must be seeded distinctly, not all from the same base_seed."""
    streams = _rng_streams(get_rngs(4, 42))
    assert any(stream != streams[0] for stream in streams[1:])


def test_get_rngs_different_seeds_differ():
    """Different base seeds must produce different streams."""
    assert _rng_streams(get_rngs(4, 42)) != _rng_streams(get_rngs(4, 7))


def test_get_rngs_unseeded_is_nondeterministic():
    """base_seed=None falls back to system entropy, so streams should not repeat."""
    assert _rng_streams(get_rngs(4, None)) != _rng_streams(get_rngs(4, None))


def test_get_rngs_rejects_non_positive_count():
    """num < 1 is invalid input."""
    with pytest.raises(AssertionError, match="num must be >= 1"):
        get_rngs(0, 42)
