# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Isaac CAP cable-routing task."""

from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _test_cable_routing_preserves_success_parameters_and_timeout(_simulation_app):
    from unittest.mock import Mock

    from isaaclab_arena.assets.cable import Cable
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena_environments.isaac_cap.cable_routing.task import CableRoutingTask, cable_route_success

    cable = Mock(spec=Cable)
    cable.name = "routing_cable"
    task = CableRoutingTask(
        cable=cable,
        pegs=[SimpleNamespace(name=f"peg_{index}") for index in range(3)],
        route_peg_indices=(2, 0),
        route_directions=(-1.0, 1.0),
        task_description="Route the cable around the last and first pegs.",
        episode_length_s=45.0,
    )
    termination_cfg = task.get_termination_cfg()
    assert isinstance(termination_cfg, TaskTerminationCfg)
    assert termination_cfg.timeout_s == 45.0
    assert termination_cfg.failures == {}
    assert len(termination_cfg.success) == 1
    objective = termination_cfg.success[0]
    assert objective.name == "cable_routing"
    assert len(objective.predicate_sequence) == 1
    route_predicate = objective.predicate_sequence[0]
    assert route_predicate.func is cable_route_success
    assert route_predicate.keywords == {
        "cable_asset_name": "routing_cable",
        "peg_asset_names": ("peg_0", "peg_1", "peg_2"),
        "route_peg_indices": (2, 0),
        "route_directions": (-1.0, 1.0),
    }
    return True


def test_cable_routing_preserves_success_parameters_and_timeout():
    assert run_function_with_persistent_simulation_app(_test_cable_routing_preserves_success_parameters_and_timeout)
