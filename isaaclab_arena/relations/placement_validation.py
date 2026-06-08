# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlacementValidationChecklist:
    """A checklist of each validation check result for placement layouts"""

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
    def rank_required_and_optional_failures(self) -> tuple[int, int]:
        """Ranking number of failed checks: required checks first, then optional checks.
        The lower the number, the better."""
        required = self._required()
        required_failed = sum(1 for item, passed in self.checklist_items.items() if item in required and not passed)
        optional_failed = sum(1 for item, passed in self.checklist_items.items() if item not in required and not passed)
        return (required_failed, optional_failed)

    def add_checklist_item(self, item: str, value: bool) -> None:
        """Add a checklist item."""
        assert item not in self.checklist_items, f"'{item}' already exists in checklist."
        self.checklist_items[item] = value
