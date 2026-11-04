# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate additional graph events without replacing native reset terms."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from isaaclab_arena.environment_spec.event_spec import EventSpec


def _graph_event(env, env_ids):
    pass


@pytest.mark.parametrize("interval", [None, [-1, 0], [2, 1], [0, float("inf")]])
def test_invalid_event_interval(interval):
    with pytest.raises(ValidationError):
        EventSpec(func="module:callable", mode="interval", interval_range_s=interval)


def test_event_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EventSpec(func="module:callable", mode="reset", interval_range=[0, 1])


def _test_events_preserve_resets_and_reject_collisions(simulation_app):
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.environment_spec.event_spec import add_event_terms

    reset = EventTermCfg(func=lambda env, env_ids: None, mode="reset")
    cfg = SimpleNamespace(events=SimpleNamespace(reset_button_state=reset))
    spec = EventSpec(
        func="isaaclab_arena.tests.test_graph_events:_graph_event",
        mode="interval",
        interval_range_s=(0, 0),
    )
    add_event_terms(cfg, {"screen": spec})
    assert cfg.events.reset_button_state is reset
    assert cfg.events.screen.func is _graph_event
    with pytest.raises(AssertionError, match="already exists"):
        add_event_terms(cfg, {"reset_button_state": spec})
    return True


def test_events_preserve_resets_and_reject_collisions():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_events_preserve_resets_and_reject_collisions)
