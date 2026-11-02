# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pool validation summary that decides the validator's exit status."""

from isaaclab_arena.offline_placement.pool_validation import print_validation_results
from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults


def _layout(passes_checks: bool, settled: bool) -> PlacementValidationResults:
    # physics_settled is left out of the required checks, so only the settled count can catch an unsettled layout.
    return PlacementValidationResults(
        validation_results={PlacementCheck.ON_RELATION: passes_checks, PlacementCheck.PHYSICS_SETTLED: settled},
        required_checks={PlacementCheck.ON_RELATION},
    )


def test_summary_passes_only_when_every_layout_passes_and_settles():
    """An unsettled layout fails the run even though it passes every required check."""
    assert print_validation_results([(0, 0, _layout(True, True)), (1, 0, _layout(True, True))])
    assert not print_validation_results([(0, 0, _layout(True, True)), (1, 0, _layout(True, False))])
    assert not print_validation_results([(0, 0, _layout(False, True))])
    assert not print_validation_results([])
