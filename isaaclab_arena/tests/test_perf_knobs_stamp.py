# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""perf_knobs stamping in record_gap_provenance (cap perf-knob provenance).

The stamp resolves every pinned performance knob from THIS worker process
environment via the Isaac-cap contract authority, re-validates the stamped
object, and fails closed before publishing so the term never emits provenance
its own validator would reject.
"""

from __future__ import annotations

import types

import pytest

from isaaclab_arena.recording.common_terms import record_gap_provenance

_eval_contract = pytest.importorskip("isaac_cap.eval_contract")
PERF_KNOB_REGISTRY = _eval_contract.PERF_KNOB_REGISTRY
ContractError = _eval_contract.ContractError


def _env(placement_seed: int = 71, seed: int = 42):
    return types.SimpleNamespace(cfg=types.SimpleNamespace(placement_seed=placement_seed, seed=seed))


def test_stamp_adds_complete_perf_knobs(monkeypatch):
    for name in PERF_KNOB_REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GAP_STREAM_TRAJECTORY", "1")
    record = record_gap_provenance(_env(), 0, provenance={"profile": "droid"})
    perf_knobs = record["gap_provenance"]["perf_knobs"]
    assert set(perf_knobs) == set(PERF_KNOB_REGISTRY)
    assert perf_knobs["GAP_STREAM_TRAJECTORY"] == "1"
    # Existing provenance and the seeds are preserved alongside the new key.
    assert record["gap_provenance"]["profile"] == "droid"
    assert record["gap_provenance"]["placement_seed"] == 71
    assert record["gap_provenance"]["seed"] == 42


def test_stamp_defaults_are_complete_not_sparse(monkeypatch):
    for name in PERF_KNOB_REGISTRY:
        monkeypatch.delenv(name, raising=False)
    record = record_gap_provenance(_env(), 0, provenance={"profile": "droid"})
    perf_knobs = record["gap_provenance"]["perf_knobs"]
    assert perf_knobs == {name: "0" for name in PERF_KNOB_REGISTRY}


def test_stamp_fails_closed_on_unregistered_perf_knob(monkeypatch):
    for name in PERF_KNOB_REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GAP_OBSERVATION_STREAM_HZ", "30")
    with pytest.raises(ContractError, match="unregistered performance-knob"):
        record_gap_provenance(_env(), 0, provenance={"profile": "droid"})


def test_stamp_fails_closed_on_illegal_gate_value(monkeypatch):
    for name in PERF_KNOB_REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAP_VLM_PAIR_CACHE", "true")
    with pytest.raises(ContractError, match="non-gate value"):
        record_gap_provenance(_env(), 0, provenance={"profile": "droid"})


def test_empty_provenance_is_untouched(monkeypatch):
    monkeypatch.setenv("GAP_STREAM_TRAJECTORY", "1")
    assert record_gap_provenance(_env(), 0, provenance=None) == {}
    assert record_gap_provenance(_env(), 0, provenance={}) == {}
