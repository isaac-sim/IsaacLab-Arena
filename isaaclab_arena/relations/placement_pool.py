# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PlacementPool:
    """Pre-solved pool of placement layouts for fast reset-time drawing.

    At build time, solves ``pool_size`` layouts in one batched call. On reset,
    each environment draws a random layout from the pool instead of re-solving.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to pre-solve.
    """

    def __init__(
        self,
        objects: Sequence[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
    ) -> None:
        self._objects = list(objects)
        self._pool: list[PlacementResult] = []

        placer = ObjectPlacer(params=placer_params)
        with torch.inference_mode(False):
            result = placer.place(self._objects, num_envs=pool_size, result_per_env=True)

        if isinstance(result, MultiEnvPlacementResult):
            self._pool = result.results
        else:
            self._pool = [result]

        n_valid = sum(1 for r in self._pool if r.success)
        n_total = len(self._pool)
        if n_valid < n_total:
            print(
                f"[WARNING] Placement pool: {n_valid}/{n_total} layouts passed validation. Using best-effort for the"
                " rest."
            )

    def draw(self, n: int) -> list[PlacementResult]:
        """Draw ``n`` layouts from the pool (random with replacement)."""
        return random.choices(self._pool, k=n)

    @property
    def size(self) -> int:
        """Number of pre-solved layouts in the pool."""
        return len(self._pool)
