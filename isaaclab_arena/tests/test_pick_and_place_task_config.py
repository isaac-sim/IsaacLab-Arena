# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration-level tests for rigid and deformable pick-and-place tasks."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_pick_and_place_task_config(simulation_app, case: str, object_type_name: str | None = None) -> bool:
    """Run one config assertion after SimulationApp initialization."""
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.spatial import (
        object_is_above_height,
        object_is_below_height,
        object_on_destination,
        object_supported_by,
        objects_in_proximity,
    )
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    class TestObject(ObjectBase):
        """Minimal object used to inspect task configs without loading USD assets."""

        def __init__(self, name: str, object_type: ObjectType):
            super().__init__(name=name, object_type=object_type)

        def get_object_pose(self, env, is_relative: bool = True):
            raise NotImplementedError

        def set_object_pose(self, env, pose, env_ids=None) -> None:
            raise NotImplementedError

        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            return AxisAlignedBoundingBox(min_point=(-0.1, -0.1, -0.1), max_point=(0.1, 0.1, 0.1))

    def make_task(
        object_type: ObjectType,
        *,
        destination_object: Asset | None = None,
        max_separation: tuple[float, float, float] | None = None,
    ) -> PickAndPlaceTask:
        pick_up_object = TestObject("pick_up", object_type)
        destination = TestObject("destination", ObjectType.BASE)
        background = Asset("background")
        background.object_min_z = -0.2
        return PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination,
            destination_object=destination_object,
            background_scene=background,
            max_separation=max_separation,
        )

    def progress_predicates(task: PickAndPlaceTask):
        objective = task.get_progress_objectives()[0]
        assert len(objective.group_names) == 1
        return [predicate for predicate, _ in objective.get_chain(objective.group_names[0])]

    object_type = ObjectType[object_type_name] if object_type_name is not None else None

    if case == "rigid":
        task = make_task(ObjectType.RIGID)
        assert task.contact_sensor_name == "contact_sensor_pick_up"
        assert getattr(task.get_scene_cfg(), task.contact_sensor_name) is not None
        success_predicates = task.get_termination_cfg().success.params["predicates"]
        assert len(success_predicates) == 1
        assert success_predicates[0].func is object_on_destination
        assert success_predicates[0].params["force_threshold"] == task.force_threshold
        assert success_predicates[0].params["velocity_threshold"] == task.velocity_threshold
        assert task.get_termination_cfg().object_dropped.func is object_is_below_height
        assert [type(metric) for metric in task.get_metrics()] == [SuccessRateMetric, ObjectMovedRateMetric]
    elif case == "deformable":
        task = make_task(ObjectType.DEFORMABLE)
        assert task.contact_sensor_name is None
        assert task.get_scene_cfg() is None
        success_predicates = task.get_termination_cfg().success.params["predicates"]
        assert len(success_predicates) == 1
        assert success_predicates[0].func is object_supported_by
        assert success_predicates[0].params["object_cfg"].name == "pick_up"
        assert success_predicates[0].params["destination_cfg"].name == "destination"
        assert task.get_termination_cfg().object_dropped.func is object_is_below_height
        assert [type(metric) for metric in task.get_metrics()] == [SuccessRateMetric]
    elif case == "unsupported":
        assert object_type is not None
        with pytest.raises(ValueError, match=f"does not support pick-up object type.*{object_type.value}"):
            make_task(object_type)
    elif case == "progress":
        assert object_type is not None
        placement_predicate = object_on_destination if object_type is ObjectType.RIGID else object_supported_by
        predicates = progress_predicates(make_task(object_type))
        assert [predicate.func for predicate in predicates] == [
            objects_settled,
            object_is_above_height,
            placement_predicate,
        ]
    elif case == "max_separation":
        assert object_type is not None
        task = make_task(object_type, max_separation=(0.1, 0.2, 0.3))
        success_predicates = task.get_termination_cfg().success.params["predicates"]
        assert success_predicates[-1].func is objects_in_proximity
        assert success_predicates[-1].params["max_x_separation"] == 0.1
        assert success_predicates[-1].params["max_y_separation"] == 0.2
        assert success_predicates[-1].params["max_z_separation"] == 0.3
        assert progress_predicates(task)[-1].func is objects_in_proximity
    elif case == "deformable_mimic":
        task = make_task(ObjectType.DEFORMABLE)
        task.mimic_env_cfg_factory = lambda _: pytest.fail("deformable Mimic factory must not run")
        with pytest.raises(NotImplementedError, match="not supported for deformable"):
            task.get_mimic_env_cfg(ArmMode.SINGLE_ARM)
    elif case == "mimic_destination":
        mimic_cfg = make_task(ObjectType.RIGID).get_mimic_env_cfg(ArmMode.SINGLE_ARM)
        assert mimic_cfg.pick_up_object_name == "pick_up"
        assert mimic_cfg.destination_location_name == "destination"
    elif case == "mimic_alias":
        destination_object = Asset("destination_alias")
        task = make_task(ObjectType.RIGID, destination_object=destination_object)
        assert task.get_mimic_env_cfg(ArmMode.SINGLE_ARM).destination_location_name == "destination_alias"
    else:
        raise ValueError(f"Unknown test case {case!r}")
    return True


def _run_case(case: str, object_type_name: str | None = None) -> None:
    result = run_function_with_persistent_simulation_app(
        _test_pick_and_place_task_config,
        case=case,
        object_type_name=object_type_name,
    )
    assert result, f"PickAndPlaceTask config case {case!r} failed"


def test_rigid_task_uses_contact_sensor_and_preserves_configs() -> None:
    _run_case("rigid")


def test_deformable_task_uses_support_without_contact_sensor() -> None:
    _run_case("deformable")


@pytest.mark.parametrize("object_type_name", ["BASE", "ARTICULATION"])
def test_unsupported_pick_up_object_type_fails_clearly(object_type_name: str) -> None:
    _run_case("unsupported", object_type_name)


@pytest.mark.parametrize("object_type_name", ["RIGID", "DEFORMABLE"])
def test_progress_uses_selected_placement_predicate(object_type_name: str) -> None:
    _run_case("progress", object_type_name)


@pytest.mark.parametrize("object_type_name", ["RIGID", "DEFORMABLE"])
def test_max_separation_remains_an_additional_success_and_progress_predicate(
    object_type_name: str,
) -> None:
    _run_case("max_separation", object_type_name)


def test_deformable_mimic_fails_before_custom_factory_runs() -> None:
    _run_case("deformable_mimic")


def test_rigid_mimic_uses_destination_location_when_alias_is_absent() -> None:
    _run_case("mimic_destination")


def test_rigid_mimic_preserves_destination_object_alias() -> None:
    _run_case("mimic_alias")
