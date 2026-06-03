# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaaclab_arena.tasks.close_door_task import CloseDoorTask
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition


def test_pick_and_place_relocates_into_destination():
    transition = PickAndPlaceTask.success_state_transition({"pick_up_object": "cube", "destination_location": "bowl"})
    # `on` (not `in`): under the AABB object solver, placing on a container == falling in it.
    assert transition == TaskTransition(
        subject="cube",
        effects=(Relocate(subject="cube", relation="on", target="bowl"),),
    )
    # Once placed, the post-success reach target is where the object ended up.
    assert transition.reach_target_on_success == "bowl"


@pytest.mark.parametrize("task_cls", [OpenDoorTask, CloseDoorTask])
def test_door_records_reach_but_no_graph_effect_yet(task_cls):
    transition = task_cls.success_state_transition({"openable_object": "microwave"})
    assert transition.subject == "microwave"
    assert transition.effects == ()  # openness is not yet representable in the graph
    assert transition.reach_target_on_success == "microwave"


def test_task_without_a_transition_raises():
    with pytest.raises(NotImplementedError, match="success_state_transition not implemented"):
        NoTask.success_state_transition({})
