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
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults


@dataclass
class PlacementCandidateBatch:
    """N candidate layouts with matching geometry, identities and optional solve results."""

    positions: list[dict[PlaceableAsset, tuple[float, float, float]]]
    """N maps of object origins in the local environment frame, in metres."""
    orientations: list[dict[PlaceableAsset, float]]
    """N maps of absolute world-Z headings in radians; absent objects retain their marker rotation."""
    bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]]
    """N maps of oriented bounds relative to object origins; each min/max tensor has shape (1, 3)."""
    env_ids: list[int]
    """N environment IDs, preserved when selecting or reordering candidates."""
    candidate_ids: list[int]
    """N original candidate IDs within their environments."""
    losses: list[float] | None = None
    """N final solver losses, or None before solving."""
    validations: list[PlacementValidationResults] | None = None
    """N check results, or None before validation."""

    def __post_init__(self) -> None:
        count = len(self.positions)
        assert len(self.orientations) == count, "One orientation map is required per candidate"
        assert len(self.bboxes) == count, "One bounding-box map is required per candidate"
        assert len(self.env_ids) == count, "One environment ID is required per candidate"
        assert len(self.candidate_ids) == count, "One candidate ID is required per candidate"
        assert self.losses is None or len(self.losses) == count, "One loss is required per solved candidate"
        assert (
            self.validations is None or len(self.validations) == count
        ), "One verdict is required per checked candidate"

    def __len__(self) -> int:
        return len(self.positions)

    def select(self, indices: list[int]) -> PlacementCandidateBatch:
        """Select rows in the requested order, retaining their original identities and results."""
        return PlacementCandidateBatch(
            positions=[self.positions[i] for i in indices],
            orientations=[self.orientations[i] for i in indices],
            bboxes=[self.bboxes[i] for i in indices],
            env_ids=[self.env_ids[i] for i in indices],
            candidate_ids=[self.candidate_ids[i] for i in indices],
            losses=None if self.losses is None else [self.losses[i] for i in indices],
            validations=None if self.validations is None else [self.validations[i] for i in indices],
        )

    def stacked_bboxes(self) -> dict[PlaceableAsset, AxisAlignedBoundingBox]:
        """Return per-object min/max bounds with shape (N, 3), in candidate order."""
        assert self.bboxes, "Cannot stack an empty candidate batch"
        return {
            obj: AxisAlignedBoundingBox(
                torch.cat([bounds[obj].min_point for bounds in self.bboxes]),
                torch.cat([bounds[obj].max_point for bounds in self.bboxes]),
            )
            for obj in self.bboxes[0]
        }
