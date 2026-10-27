# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math
import random
import torch


def derive_seed(*key_parts: object) -> int:
    """Derive an independent 64-bit torch seed from key parts such as ``(seed, env_id, candidate_id)``.

    Unlike ``seed + offset``, nearby keys never share a seed: ``derive_seed(42, 1)`` and ``derive_seed(43, 0)`` differ.
    """
    digest = hashlib.sha256(":".join(str(part) for part in key_parts).encode()).digest()
    # Torch generator seeds are 64-bit, so keep the first 8 bytes of the digest.
    return int.from_bytes(digest[:8], "big")


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
