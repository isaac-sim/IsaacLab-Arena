# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PlacementResult."""

from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults


def _make_result(
    *,
    required: dict[PlacementCheck, bool] | None = None,
    optional: dict[PlacementCheck, bool] | None = None,
    final_loss: float = 0.0,
) -> PlacementResult:
    required = required or {PlacementCheck.NO_OVERLAP: True}
    all_checks = dict(required)
    if optional:
        all_checks.update(optional)
    return PlacementResult(
        validation_results=PlacementValidationResults(
            validation_results=all_checks,
            required_checks=set(required),
        ),
        positions={},
        final_loss=final_loss,
        attempts=1,
    )


def test_success_true_when_all_required_pass():
    result = _make_result(required={PlacementCheck.NO_OVERLAP: True, PlacementCheck.ON_RELATION: True})
    assert result.success is True


def test_success_false_when_required_fails():
    result = _make_result(required={PlacementCheck.NO_OVERLAP: False})
    assert result.success is False


def test_success_ignores_failed_optional_check():
    result = _make_result(
        required={PlacementCheck.NO_OVERLAP: True},
        optional={PlacementCheck.PHYSICS_SETTLED: False},
    )
    assert result.success is True
