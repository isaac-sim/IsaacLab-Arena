# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for episode outcome classification (no Isaac Sim required)."""

from isaaclab_arena.evaluation.episode_outcome import classify_outcome


def test_success_term_maps_to_success():
    assert classify_outcome("success") == "success"


def test_no_term_means_timeout():
    assert classify_outcome(None) == "timeout"


def test_other_term_means_failure():
    assert classify_outcome("object_dropped") == "failure"
