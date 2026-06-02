# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab_arena.utils.random import get_random_rotation


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
