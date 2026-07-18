# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class PlacementCheck(StrEnum):

    NO_OVERLAP = "no_overlap"
    """Build-time check: no two placed object bounding boxes intersect."""

    ON_RELATION = "on_relation"
    """Build-time check: every ``On`` relation holds — the child rests on its parent within the
    configured Z tolerance."""

    NEXT_TO = "next_to"
    """Build-time check: every ``NextTo`` relation holds — child on the requested side at the target
    offset, within the relation's ``tolerance_m``."""

    NOT_NEXT_TO = "not_next_to"
    """Build-time check: every ``NotNextTo`` relation holds — child has cleared the keep-out zone
    beside the parent, within the relation's ``tolerance_m``."""

    FACE_TO = "face_to"
    """Build-time check: every ``FaceTo`` subject has a defined target direction and facing yaw."""

    PHYSICS_SETTLED = "physics_settled"
    """Run-time check: after stepping physics the movable objects' velocities fall
    below threshold, i.e. the layout is stable and does not drift or topple."""

    IK_REACHABLE = "ik_reachable"
    """Run-time check: the robot can reach a top-down grasp at every movable object's
    resting pose, per cuRobo's collision-aware IK solver."""


@dataclass
class PlacementValidationResults:
    """A collection of validation check results for placement layouts.

    Keys are check names (see :class:`PlacementCheck` for the standard set and what each check means).
    """

    validation_results: dict[str, bool] = field(default_factory=dict)

    required_checks: set[str] = field(default_factory=set)
    """Names of checks that must pass for the layout to be valid. Empty means every check that ran is required."""

    def _required(self) -> set[str]:
        """Required check names, defaulting to every check that produced a result when none are declared."""
        return self.required_checks or set(self.validation_results)

    def do_all_required_validation_checks_pass(self, required_checks: list[str] | None = None) -> bool:
        """Check whether all the required validation checks pass.

        Args:
            required_checks: checks that must pass. If None, only required checks gate (optional checks are ignored).
        Returns:
            True if all the required checks pass, False otherwise
        """
        if required_checks is None:
            required_checks = self._required()
        return all(self.validation_results.get(check, True) for check in required_checks)

    @property
    def get_failed_validation_check_names(self) -> list[str]:
        """Get the failed validation check names."""
        return [str(check) for check, passed in self.validation_results.items() if not passed]

    def report(self) -> str:
        """One-line report check items and their results."""
        verdict = "PASS" if self.do_all_required_validation_checks_pass() else "FAIL"
        checks = ", ".join(f"{check}={passed}" for check, passed in self.validation_results.items()) or "no checks"
        return f"{verdict} [{checks}]"

    @property
    def get_number_of_required_and_optional_failures(self) -> tuple[int, int]:
        """Get the number of required and optional validation checks that failed."""
        required = self._required()
        failed = [check for check, passed in self.validation_results.items() if not passed]
        required_failed = sum(1 for check in failed if check in required)
        optional_failed = sum(1 for check in failed if check not in required)
        return (required_failed, optional_failed)

    def add_validation_check(self, check: str, value: bool, required: bool = False) -> None:
        """Add a validation check.

        Args:
            check: Check name; must not already exist.
            value: Whether the check passed.
            required: If True, the check also gates do_all_required_validation_checks_pass() (and thus success).
        """
        assert check not in self.validation_results, f"'{check}' already exists in validation results."
        self.validation_results[check] = value
        if required:
            self.required_checks.add(check)
