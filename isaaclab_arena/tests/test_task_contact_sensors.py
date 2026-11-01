# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contact sensors that tasks add to the scene configuration."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_task_contact_sensors_follow_object_spawn(simulation_app, tmp_path, task_name) -> bool:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.object_in_task import ObjectInTask
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
    from isaaclab_arena.tests.test_object_set import _write_rigid_cube_usd

    def make_rigid_cuboid(name):
        """Return a rigid cuboid object, whose rigid body is at its spawn root."""
        cuboid_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), rigid_props=sim_utils.RigidBodyPropertiesCfg())
        return Object(name=name, object_type=ObjectType.RIGID, spawner_cfg=cuboid_cfg)

    pair_count = 2 if task_name == "sorting" else 1
    pickups = [make_rigid_cuboid(f"pickup_{pair_index}") for pair_index in range(pair_count)]
    destinations = [make_rigid_cuboid(f"destination_{pair_index}") for pair_index in range(pair_count)]
    # The tasks read the background only in get_termination_cfg(), which this test does not call.
    background = destinations[0]
    if task_name == "pick_and_place":
        task = PickAndPlaceTask(pickups[0], destinations[0], background)
        sensor_names = [task.contact_sensor_name]
    elif task_name == "object_in":
        task = ObjectInTask(pickups[0], destinations[0])
        sensor_names = [task.contact_sensor_cfg.name]
    else:
        task = SortMultiObjectTask(pickups, destinations, background)
        sensor_names = task.contact_sensor_name_list

    # Replace each spawn configuration after the task is built, as Object.set_spawned_asset() does. The file's
    # default prim is /Object, so its rigid body is at /Group/Body below the spawn root.
    nested_body_usd_path = _write_rigid_cube_usd(tmp_path / "nested_body.usda", body_path="/Object/Group/Body")
    for scene_object in pickups + destinations:
        nested_body_spawn_cfg = sim_utils.UsdFileCfg(usd_path=nested_body_usd_path, activate_contact_sensors=True)
        scene_object.object_cfg.spawn = nested_body_spawn_cfg

    scene_cfg = task.get_scene_cfg()
    for sensor_name, pickup, destination in zip(sensor_names, pickups, destinations, strict=True):
        sensor_cfg = getattr(scene_cfg, sensor_name)
        assert sensor_cfg.prim_path == f"{pickup.prim_path}/Group/Body"
        assert sensor_cfg.filter_prim_paths_expr == [f"{destination.prim_path}/Group/Body"]
    return True


@pytest.mark.parametrize("task_name", ["pick_and_place", "object_in", "sorting"])
def test_task_contact_sensors_follow_object_spawn(tmp_path, task_name):
    assert run_function_with_persistent_simulation_app(
        _test_task_contact_sensors_follow_object_spawn, tmp_path=tmp_path, task_name=task_name
    )
