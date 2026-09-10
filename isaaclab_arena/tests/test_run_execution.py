# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for run_execution's datagen recorder-term wiring.

Importing run_execution needs a running SimulationApp (it imports RecorderManagerBaseCfg
from isaaclab.managers.recorder_manager) -- see isaaclab_arena/tests/test_task_registry.py
for the established _test_/test_ + run_function_with_persistent_simulation_app pattern
this mirrors.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_merges_datagen_term_into_none_recorders_cfg(simulation_app):
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.evaluation.run_execution import _with_datagen_recorder_term
    from isaaclab_arena.recording.callback_recorder_term import CallbackRecorderTermHandlers

    merged = _with_datagen_recorder_term(None, build_handlers=lambda env: CallbackRecorderTermHandlers())

    assert isinstance(merged, RecorderManagerBaseCfg)
    assert hasattr(merged, "datagen_callback")
    return True


def test_merges_datagen_term_into_none_recorders_cfg():
    assert run_function_with_persistent_simulation_app(_test_merges_datagen_term_into_none_recorders_cfg)


def _test_preserves_existing_recorder_terms_alongside_datagen_term(simulation_app):
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.evaluation.run_execution import _with_datagen_recorder_term
    from isaaclab_arena.recording.callback_recorder_term import CallbackRecorderTermHandlers
    from isaaclab_arena.utils.configclass import make_configclass

    ExistingRecordersCfg = make_configclass(
        "ExistingRecordersCfg", [("existing_term", object, "sentinel")], bases=(RecorderManagerBaseCfg,)
    )
    existing = ExistingRecordersCfg()

    merged = _with_datagen_recorder_term(existing, build_handlers=lambda env: CallbackRecorderTermHandlers())

    assert merged.existing_term == "sentinel"
    assert hasattr(merged, "datagen_callback")
    return True


def test_preserves_existing_recorder_terms_alongside_datagen_term():
    assert run_function_with_persistent_simulation_app(_test_preserves_existing_recorder_terms_alongside_datagen_term)
