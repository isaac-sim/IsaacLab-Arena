# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import pytest

# Task classes are imported lazily inside each test. Importing them pulls in isaaclab -> omni/pxr,
# and doing that at module-import (pytest collection) time would load pxr before any SimulationApp
# starts -- which breaks SimulationApp.startup() for every sim test in the session.
# ``task_transition`` is pure Python, so it is safe to import at the top level.
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition


def test_pick_and_place_relocates_into_destination():
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    transition = PickAndPlaceTask.success_state_transition(pick_up_object="cube", destination_location="bowl")
    # `on` (not `in`): under the AABB object solver, placing on a container == falling in it.
    assert transition == TaskTransition(
        subject="cube",
        effects=(Relocate(subject="cube", relation="on", target="bowl"),),
    )
    # Once placed, the post-success reach target is where the object ended up.
    assert transition.reach_target_on_success == "bowl"


@pytest.mark.parametrize("task_module_name", ["open_door_task", "close_door_task"])
def test_door_records_reach_but_no_graph_effect_yet(task_module_name):
    # Each module registers exactly one Rotate*-derived door task; resolve it by class-name convention.
    module = importlib.import_module(f"isaaclab_arena.tasks.{task_module_name}")
    class_name = "".join(part.capitalize() for part in task_module_name.split("_"))
    task_cls = getattr(module, class_name)

    transition = task_cls.success_state_transition(openable_object="microwave")
    assert transition.subject == "microwave"
    assert transition.effects == ()  # openness is not yet representable in the graph
    assert transition.reach_target_on_success == "microwave"


def test_task_without_a_transition_raises():
    from isaaclab_arena.tasks.no_task import NoTask

    with pytest.raises(NotImplementedError, match="success_state_transition not implemented"):
        NoTask.success_state_transition()
