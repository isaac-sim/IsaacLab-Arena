# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_task_registry_resolves_concrete_tasks(simulation_app):
    from isaaclab_arena.assets.registries import TaskRegistry

    expected = [
        "NoTask",
        "PickAndPlaceTask",
        "PlaceUprightTask",
        "LiftObjectTask",
        "LiftObjectTaskRL",
        "DexsuiteLiftTask",
        "GoalPoseTask",
        "PressButtonTask",
        "RotateRevoluteJointTask",
        "CloseDoorTask",
        "OpenDoorTask",
        "TurnKnobTask",
        "AssemblyTask",
        "SortMultiObjectTask",
    ]
    registry = TaskRegistry()
    for name in expected:
        cls = registry.get_task_by_name(name)
        assert cls.__name__ == name, f"{name} -> {cls.__name__}"
    return True


def test_task_registry_resolves_concrete_tasks():
    result = run_simulation_app_function(_test_task_registry_resolves_concrete_tasks)
    assert result
