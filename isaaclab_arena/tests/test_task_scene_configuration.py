# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task contact sensors follow the current source and destination assets."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_rigid_asset(path, body_suffix):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body_path = f"/Asset{body_suffix}"
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    cube = UsdGeom.Cube.Define(stage, f"{body_path}/Cube")
    cube.CreateSizeAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()


def _make_task(task_kind, pickups, destinations):
    from isaaclab_arena.tasks.object_in_task import ObjectInTask
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask

    background = SimpleNamespace(object_min_z=-1.0)
    if task_kind == "pick_and_place":
        return PickAndPlaceTask(pickups[0], destinations[0], background)
    if task_kind == "object_in":
        return ObjectInTask(pickups[0], destinations[0])
    assert task_kind == "sorting"
    return SortMultiObjectTask(pickups, destinations, background)


def _termination_sensor_names(task, task_kind):
    success_sequence = task.get_termination_cfg().success[0].predicate_sequence
    if task_kind == "sorting":
        predicates = success_sequence[0].keywords["predicates"]
        return [predicate.params["contact_sensor_cfg"].name for predicate in predicates]
    return [success_sequence[-1].predicate.keywords["contact_sensor_cfg"].name]


def _test_contact_sensors_follow_current_assets(simulation_app, tmp_path, task_kind, object_count):
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    pickups = []
    destinations = []
    sensor_names = []
    for object_index in range(object_count):
        pickup = Object(
            name=f"pickup_{object_index}",
            object_type=ObjectType.RIGID,
            spawn_cfg=UsdFileCfg(usd_path=str(tmp_path / f"pickup_{object_index}_0.usda")),
        )
        destination = Object(
            name=f"destination_{object_index}",
            object_type=ObjectType.RIGID,
            spawn_cfg=UsdFileCfg(usd_path=str(tmp_path / f"destination_{object_index}_0.usda")),
        )
        pickups.append(pickup)
        destinations.append(destination)
        sensor_name = f"contact_sensor_{pickup.name}"
        if task_kind == "object_in":
            sensor_name += f"_in_{destination.name}"
        sensor_names.append(sensor_name)

    # Source files need not exist until the builder requests task scene configuration.
    task = _make_task(task_kind, pickups, destinations)
    assert _termination_sensor_names(task, task_kind) == sensor_names

    body_suffixes = [
        ("", "/OriginalTarget"),
        ("/Group/UpdatedBody", "/Group/UpdatedTarget"),
        ("/FinalBody", ""),
    ]
    for revision, (pickup_suffix, destination_suffix) in enumerate(body_suffixes):
        for pickup, destination in zip(pickups, destinations, strict=True):
            pickup_path = tmp_path / f"{pickup.name}_{revision}.usda"
            destination_path = tmp_path / f"{destination.name}_{revision}.usda"
            _write_rigid_asset(pickup_path, pickup_suffix)
            _write_rigid_asset(destination_path, destination_suffix)
            pickup.spawn_cfg.usd_path = str(pickup_path)
            destination.spawn_cfg.usd_path = str(destination_path)

        scene_cfg = task.get_scene_cfg()
        for sensor_name, pickup, destination in zip(sensor_names, pickups, destinations, strict=True):
            sensor_cfg = getattr(scene_cfg, sensor_name)
            assert sensor_cfg.prim_path == pickup.get_prim_path() + pickup_suffix
            assert sensor_cfg.filter_prim_paths_expr == [destination.get_prim_path() + destination_suffix]
        assert _termination_sensor_names(task, task_kind) == sensor_names
    return True


@pytest.mark.parametrize("task_kind,object_count", [("pick_and_place", 1), ("object_in", 1), ("sorting", 2)])
def test_contact_sensors_follow_current_assets(tmp_path, task_kind, object_count):
    assert run_function_with_persistent_simulation_app(
        _test_contact_sensors_follow_current_assets,
        tmp_path=tmp_path,
        task_kind=task_kind,
        object_count=object_count,
    )


def _test_deformable_pickup_omits_contact_sensor(simulation_app):
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    pickup = SimpleNamespace(
        name="pickup",
        object_type=ObjectType.DEFORMABLE,
        get_contact_sensor_cfg=Mock(side_effect=AssertionError("Deformable pickup cannot use a contact sensor")),
    )
    destination = SimpleNamespace(name="destination", object_type=ObjectType.RIGID)
    task = PickAndPlaceTask(pickup, destination, SimpleNamespace(object_min_z=-1.0))
    assert task.get_scene_cfg() is None
    assert task.contact_sensor_name is None
    placement_requirement = task.get_termination_cfg().success[0].predicate_sequence[-1]
    assert placement_requirement.predicate.keywords["contact_sensor_cfg"] is None
    pickup.get_contact_sensor_cfg.assert_not_called()
    return True


def test_deformable_pickup_omits_contact_sensor():
    assert run_function_with_persistent_simulation_app(_test_deformable_pickup_omits_contact_sensor)


def _test_composite_collects_each_scene_config_once(simulation_app):
    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.utils.configclass import make_configclass

    subtasks = []
    sensor_paths = {
        "first_contacts": "/World/First",
        "second_contacts": "/World/Second",
    }
    for sensor_name, prim_path in sensor_paths.items():
        scene_cfg = make_configclass(
            "SceneCfg",
            [(sensor_name, ContactSensorCfg, ContactSensorCfg(prim_path=prim_path))],
        )()
        subtasks.append(SimpleNamespace(get_scene_cfg=Mock(return_value=scene_cfg)))
    subtasks.append(SimpleNamespace(get_scene_cfg=Mock(return_value=None)))
    task = CompositeTaskBase(subtasks, episode_length_s=10.0)

    for expected_calls in (1, 2):
        scene_cfg = task.get_scene_cfg()
        for subtask in subtasks:
            assert subtask.get_scene_cfg.call_count == expected_calls
        for sensor_name, prim_path in sensor_paths.items():
            assert getattr(scene_cfg, sensor_name).prim_path == prim_path
    return True


def test_composite_collects_each_scene_config_once():
    assert run_function_with_persistent_simulation_app(_test_composite_collects_each_scene_config_once)
