# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pick_and_place_maple_table environment."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_task_objects_are_named_by_role(simulation_app):
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    pick_up_object_name = "rubiks_cube_hot3d_robolab"
    environment_cfg = PickAndPlaceMapleTableEnvironmentCfg(pick_up_object=pick_up_object_name)
    arena_env = PickAndPlaceMapleTableEnvironment().build(environment_cfg)

    assert arena_env.scene.assets["pick_up_object"].registry_name == pick_up_object_name
    assert arena_env.scene.assets["destination_location"].registry_name == "bowl_ycb_robolab"
    # The description names the registered objects, as it did when the objects had their registry names.
    expected_description = f"Pick up the {pick_up_object_name}, and place it into the bowl_ycb_robolab"
    assert arena_env.task.get_task_description() == expected_description
    return True


def test_task_objects_are_named_by_role():
    assert run_function_with_persistent_simulation_app(_test_task_objects_are_named_by_role, headless=HEADLESS)
