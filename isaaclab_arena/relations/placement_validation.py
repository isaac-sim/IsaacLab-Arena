# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PlacementCheck(StrEnum):
    """Standard names for the placement validation checks."""

    NO_OVERLAP = "no_overlap"
    """Geometric check (sim-free): no two placed object bounding boxes intersect. Gates success."""

    ON_RELATION = "on_relation"
    """Geometric check (sim-free): every ``On`` relation holds — the child rests on its parent within the
    configured Z tolerance. Gates success."""

    PHYSICS_SETTLED = "physics_settled"
    """Dynamic check (needs the live Sim App): after stepping physics the movable objects' velocities fall
    below threshold, i.e. the layout is stable and does not drift or topple. Optional (a failure triggers re-selection rather than failing the layout outright)."""


@dataclass
class PlacementValidationChecklist:
    """A checklist of each validation check result for placement layouts.

    Keys are check names (see :class:`PlacementCheck` for the standard set and what each check means).
    """

    checklist_items: dict[str, bool] = field(default_factory=dict)

    required_items: set[str] = field(default_factory=set)
    """Names of checks that must pass for the layout to be valid. Empty means every check is required."""

    def _required(self) -> set[str]:
        """Required check names, defaulting to every check when none are declared."""
        return self.required_items or set(self.checklist_items.keys())

    def pass_validation_checklist(self, checklist_items: list[str] | None = None) -> bool:
        """Check whether the gating checks pass. Defaults to the required checks only.
        Args:
            checklist_items: checks that must pass. If None, only required checks gate (optional checks are ignored).
        Returns:
            True if all the considered checks pass, False otherwise
        """
        if checklist_items is None:
            checklist_items = self._required()
        return all(self.checklist_items[item] for item in checklist_items)

    @property
    def failed_checklist_items(self) -> list[str]:
        """Get the failed checklist items."""
        return [item for item in self.checklist_items.keys() if not self.checklist_items[item]]

    @property
    def physics_settled_failed(self) -> bool:
        """True when this layout was stepped in sim and did not settle."""
        return self.checklist_items.get(PlacementCheck.PHYSICS_SETTLED) is False

    @property
    def rank_required_and_optional_failures(self) -> tuple[int, int]:
        """Ranking number of failed checks: required checks first, then optional checks.
        The lower the number, the better."""
        required = self._required()
        required_failed = sum(1 for item, passed in self.checklist_items.items() if item in required and not passed)
        optional_failed = sum(1 for item, passed in self.checklist_items.items() if item not in required and not passed)
        return (required_failed, optional_failed)

    def add_checklist_item(self, item: str, value: bool, required: bool = False) -> None:
        """Add a checklist item.

        Args:
            item: Check name; must not already exist.
            value: Whether the check passed.
            required: If True, the item also gates pass_validation_checklist() (and thus success).
        """
        assert item not in self.checklist_items, f"'{item}' already exists in checklist."
        self.checklist_items[item] = value
        if required:
            self.required_items.add(item)
