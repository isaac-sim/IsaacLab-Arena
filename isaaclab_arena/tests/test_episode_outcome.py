# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for episode outcome classification (no Isaac Sim required)."""

from __future__ import annotations

import torch

from isaaclab_arena.evaluation.episode_outcome import classify_outcome


class _FakeTerminationManager:
    """Minimal stand-in for IsaacLab's TerminationManager."""

    def __init__(self, terms: dict[str, torch.Tensor]) -> None:
        self._terms = terms

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms)

    def get_term(self, name: str) -> torch.Tensor:
        return self._terms[name]


class _FakeEnv:
    def __init__(self, termination_manager: _FakeTerminationManager) -> None:
        self.termination_manager = termination_manager


def test_success_term_true_means_success():
    env = _FakeEnv(_FakeTerminationManager({"success": torch.tensor([True, False])}))
    assert classify_outcome(env, 0) == "success"


def test_time_out_term_true_means_timeout():
    env = _FakeEnv(_FakeTerminationManager({"success": torch.tensor([False]), "time_out": torch.tensor([True])}))
    assert classify_outcome(env, 0) == "timeout"


def test_neither_term_true_means_failure():
    env = _FakeEnv(_FakeTerminationManager({"success": torch.tensor([False]), "time_out": torch.tensor([False])}))
    assert classify_outcome(env, 0) == "failure"


def test_no_success_or_time_out_terms_means_failure():
    env = _FakeEnv(_FakeTerminationManager({"object_dropped": torch.tensor([True])}))
    assert classify_outcome(env, 0) == "failure"
