# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate layouts and their identities throughout placement."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.validation.types import PlacementValidationResults


@dataclass
class PlacementCandidate:
    """One working layout of all placement objects in one environment."""

    env_id: int
    """Environment whose geometry and object variants this layout uses."""
    candidate_id: int
    """Sample index within the environment, unchanged by filtering or ranking."""
    positions: dict[PlaceableAsset, tuple[float, float, float]]
    """Object origins in the local environment frame, in metres; each value has shape (3,)."""
    orientations: dict[PlaceableAsset, float]
    """Absolute world Z headings in radians. Missing objects retain their marker rotation."""
    bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox]
    """Bounds enclosing each object's orientation, relative to its origin; min/max tensors have shape (1, 3)."""
    loss: float | None = None
    """Final solver loss, or None before solving."""
    validation: PlacementValidationResults | None = None
    """Check outcomes, or None before validation."""


@dataclass
class PlacementCandidateBatch:
    """N complete layouts, potentially with several candidates per environment."""

    candidates: list[PlacementCandidate]
    """Layouts in batch order, each with its own identity, geometry and results."""

    def __len__(self) -> int:
        return len(self.candidates)

    def select(self, indices: list[int]) -> PlacementCandidateBatch:
        """Select or reorder layouts by batch index, retaining references to the same candidates."""
        return PlacementCandidateBatch([self.candidates[i] for i in indices])

    def stacked_bboxes(self) -> dict[PlaceableAsset, AxisAlignedBoundingBox]:
        """Stack bounds into (N, 3) tensors in candidate order."""
        assert self.candidates, "Cannot stack bounds for an empty candidate batch"
        return {
            obj: AxisAlignedBoundingBox(
                torch.cat([candidate.bboxes[obj].min_point for candidate in self.candidates]),
                torch.cat([candidate.bboxes[obj].max_point for candidate in self.candidates]),
            )
            for obj in self.candidates[0].bboxes
        }
