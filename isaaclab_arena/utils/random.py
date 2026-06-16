# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import random
import torch


def get_random_rotation(generator: torch.Generator | None = None) -> float:
    """Sample a uniform yaw in [-pi, pi) radians (rotation about Z)."""
    u = torch.rand(1, generator=generator).item()
    return (2.0 * u - 1.0) * math.pi


def get_rngs(num: int, base_seed: int | None) -> list[random.Random]:
    """Build num independent RNGs, reproducible under base_seed.

    base_seed=None falls back to system entropy (non-reproducible).
    """
    assert num >= 1, f"num must be >= 1, got {num}"
    if base_seed is None:
        return [random.Random() for _ in range(num)]
    else:
        seeder = random.Random(base_seed)
        return [random.Random(seeder.getrandbits(64)) for _ in range(num)]
