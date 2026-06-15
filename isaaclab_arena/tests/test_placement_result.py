# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified PlacementResult type."""

import pytest

from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation import PlacementCheck, PlacementValidationResults


def _make_leaf(*, success: bool = True, final_loss: float = 0.0, attempts: int = 1) -> PlacementResult:
    """Create a directly-constructed (leaf) PlacementResult for testing."""
    return PlacementResult(
        validation_results=PlacementValidationResults(
            validation_results={PlacementCheck.NO_OVERLAP: success},
            required_checks={PlacementCheck.NO_OVERLAP},
        ),
        positions={},
        final_loss=final_loss,
        attempts=attempts,
    )


def test_leaf_results_returns_self():
    """Directly-constructed result's .results returns [self]."""
    leaf = _make_leaf()
    assert leaf.results == [leaf]
    assert leaf.results[0] is leaf


def test_leaf_success_delegates_to_validation():
    passing = _make_leaf(success=True)
    assert passing.success is True

    failing = _make_leaf(success=False)
    assert failing.success is False


def test_from_per_env_results_returns_original_list():
    leaves = [_make_leaf() for _ in range(3)]
    wrapper = PlacementResult.from_per_env(leaves)
    assert len(wrapper.results) == 3
    for i, r in enumerate(wrapper.results):
        assert r is leaves[i]


def test_from_per_env_single_preserves_identity():
    """from_per_env([single]).results[0] is the original single leaf."""
    single = _make_leaf()
    wrapper = PlacementResult.from_per_env([single])
    assert wrapper.results[0] is single


def test_from_per_env_top_level_fields_mirror_first():
    """Wrapper's top-level fields come from results[0], not results[1]."""
    leaf_a = _make_leaf(final_loss=1.5, attempts=10)
    leaf_b = _make_leaf(final_loss=3.0, attempts=20)
    wrapper = PlacementResult.from_per_env([leaf_a, leaf_b])
    assert wrapper.validation_results is leaf_a.validation_results
    assert wrapper.positions is leaf_a.positions
    assert wrapper.final_loss == 1.5
    assert wrapper.attempts == 10
    assert wrapper.orientations is leaf_a.orientations


def test_success_multi_env_all_pass():
    leaves = [_make_leaf(success=True) for _ in range(4)]
    wrapper = PlacementResult.from_per_env(leaves)
    assert wrapper.success is True


def test_success_multi_env_mixed():
    """When some envs fail, wrapper.success is False."""
    leaves = [_make_leaf(success=True), _make_leaf(success=False), _make_leaf(success=True)]
    wrapper = PlacementResult.from_per_env(leaves)
    assert wrapper.success is False


def test_success_multi_env_all_fail():
    leaves = [_make_leaf(success=False) for _ in range(2)]
    wrapper = PlacementResult.from_per_env(leaves)
    assert wrapper.success is False


def test_from_per_env_empty_list_raises():
    with pytest.raises(AssertionError, match="from_per_env requires at least one result"):
        PlacementResult.from_per_env([])


def test_from_per_env_rejects_nested_wrappers():
    """Wrapping a wrapper is not supported and should raise."""
    leaf = _make_leaf()
    wrapper = PlacementResult.from_per_env([leaf])
    with pytest.raises(AssertionError, match="wrapping a wrapper is not supported"):
        PlacementResult.from_per_env([wrapper])


def test_dataclasses_replace_strips_per_env_results():
    """dataclasses.replace() does not preserve _per_env_results (init=False field).

    Rebuild via from_per_env(original.results) instead.
    """
    import dataclasses

    leaves = [_make_leaf(success=True), _make_leaf(success=False)]
    wrapper = PlacementResult.from_per_env(leaves)
    assert wrapper.success is False

    copied = dataclasses.replace(wrapper)
    assert copied._per_env_results is None
    assert copied.success is True
