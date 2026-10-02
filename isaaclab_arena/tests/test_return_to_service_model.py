# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise diagnostic rework, physical interlocks, and test provenance without simulation."""

import math
from dataclasses import replace

import pytest

from isaaclab_arena_environments.return_to_service.model import ServiceModel, ServiceSnapshot
from isaaclab_arena_environments.return_to_service.scenarios import SCENARIOS


def _start(name="healthy"):
    scenario = SCENARIOS[name]
    model = ServiceModel(scenario)
    snapshot = ServiceSnapshot(inlet_obstructed=scenario.inlet_obstruction)
    model.update(snapshot, 0.0)
    return model, snapshot


def _step(model, snapshot, dt=0.1, **changes):
    # These scripted tests fully insert/extract a battery unless they explicitly
    # exercise the interval between releasing retention and clearing the contacts.
    if "installed_battery" in changes and "battery_connected" not in changes:
        changes["battery_connected"] = changes["installed_battery"] is not None
    snapshot = replace(snapshot, **changes)
    model.update(snapshot, dt)
    return snapshot


def _load_test(model, snapshot, battery_id):
    snapshot = _step(model, snapshot, installed_battery=None, battery_in_tester=battery_id, battery_test_pressed=False)
    snapshot = _step(model, snapshot, battery_test_pressed=True)
    return _step(model, snapshot, dt=model.test_duration_s, battery_test_pressed=False)


def _airflow_test(model, snapshot, battery_id):
    snapshot = _step(
        model,
        snapshot,
        installed_battery=battery_id,
        battery_in_tester=None,
        vacuum_in_test_dock=True,
        power_on=True,
        airflow_test_pressed=False,
    )
    snapshot = _step(model, snapshot, airflow_test_pressed=True)
    return _step(model, snapshot, dt=model.test_duration_s, airflow_test_pressed=False)


def _service_airway(model, snapshot, filter_id):
    snapshot = _step(model, snapshot, installed_battery=None, power_on=False)
    snapshot = _step(model, snapshot, cup_closed=False, installed_filter=None)
    snapshot = _step(model, snapshot, inlet_obstructed=False, cup_debris_present=False, debris_in_waste=True)
    return _step(model, snapshot, installed_filter=filter_id, cup_closed=True)


def _pack(model, snapshot, battery_id):
    return _step(
        model,
        snapshot,
        installed_battery=None,
        battery_in_tester=None,
        vacuum_in_test_dock=False,
        power_on=False,
        vacuum_packed=True,
        packed_battery=battery_id,
        packed_accessories=model.scenario.requested_accessories,
        case_closed=True,
        case_latched=True,
        removed_battery_in_service=model.scenario.weak_battery,
        removed_filter_in_service=model.scenario.clogged_filter,
        obstruction_in_waste=True,
        station_reset=True,
        objects_released=True,
    )


def _finish(name="healthy"):
    model, snapshot = _start(name)
    scenario = model.scenario
    battery_id = scenario.spare_battery if scenario.weak_battery else scenario.original_battery
    filter_id = scenario.spare_filter if scenario.clogged_filter else scenario.original_filter
    snapshot = _load_test(model, snapshot, battery_id)
    snapshot = _service_airway(model, snapshot, filter_id)
    snapshot = _airflow_test(model, snapshot, battery_id)
    snapshot = _pack(model, snapshot, battery_id)
    return model, snapshot


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_all_fault_combinations_have_a_valid_service_solution(scenario_name):
    model, _ = _finish(scenario_name)
    assert model.status.success
    assert model.status.battery_test_count == 1
    assert model.status.airflow_test_count == 1
    assert model.status.unnecessary_replacements == 0


def test_combined_faults_require_rework_after_the_motor_starts():
    model, snapshot = _start("combined")
    snapshot = _airflow_test(model, snapshot, "battery_original")
    assert model.status.airflow_reading.value == 0.0
    assert not model.status.vacuum_verified
    snapshot = _load_test(model, snapshot, "battery_original")
    assert model.status.battery_reading.result == "fail"
    snapshot = _load_test(model, snapshot, "battery_spare")
    snapshot = _airflow_test(model, snapshot, "battery_spare")
    restricted_flow = model.status.airflow_reading.value
    assert 0 < restricted_flow < model.MIN_AIRFLOW_LPS
    snapshot = _step(model, snapshot, installed_battery=None, power_on=False)
    snapshot = _step(model, snapshot, cup_closed=False)
    snapshot = _step(model, snapshot, inlet_obstructed=False, cup_debris_present=False, debris_in_waste=True)
    snapshot = _step(model, snapshot, cup_closed=True)
    snapshot = _airflow_test(model, snapshot, "battery_spare")
    assert restricted_flow < model.status.airflow_reading.value < model.MIN_AIRFLOW_LPS
    snapshot = _service_airway(model, snapshot, "filter_spare")
    snapshot = _airflow_test(model, snapshot, "battery_spare")
    _pack(model, snapshot, "battery_spare")
    assert model.status.success
    assert model.status.airflow_test_count == 4


def test_same_tested_battery_can_be_removed_for_packing():
    model, snapshot = _finish()
    certificate = model.certificate
    assert model.status.success
    _step(model, snapshot, dt=5.0)
    assert model.certificate == certificate
    assert model.status.success


def test_substituting_a_battery_invalidates_test_even_after_original_is_restored():
    model, snapshot = _finish()
    snapshot = _step(model, snapshot, installed_battery="battery_spare", packed_battery=None)
    snapshot = _step(model, snapshot, installed_battery=None, packed_battery="battery_original")
    assert model.certificate is None
    assert not model.status.vacuum_verified
    assert not model.status.success
    assert model.status.unnecessary_replacements == 1


@pytest.mark.parametrize(
    "change",
    [
        {"cup_closed": False},
        {"installed_filter": None},
        {"installed_filter": "filter_spare"},
        {"inlet_obstructed": True},
        {"cup_debris_present": True},
    ],
)
def test_airway_changes_invalidate_test_permanently_until_retest(change):
    model, snapshot = _finish()
    before = snapshot
    snapshot = _step(model, snapshot, **change)
    model.update(before, 0.1)
    assert model.certificate is None
    assert not model.status.success


def test_disposal_after_an_airflow_test_preserves_the_assembly_certificate():
    model, snapshot = _start()
    snapshot = _load_test(model, snapshot, "battery_original")
    snapshot = _service_airway(model, snapshot, "filter_original")
    snapshot = _step(model, snapshot, debris_in_waste=False)
    snapshot = _airflow_test(model, snapshot, "battery_original")
    certificate = model.certificate
    assert certificate is not None
    assert model.status.airflow_reading.value == model.NOMINAL_AIRFLOW_LPS
    snapshot = _pack(model, snapshot, "battery_original")
    assert model.status.airway_serviced
    assert not model.status.success
    _step(model, snapshot, debris_in_waste=True)
    assert model.certificate == certificate
    assert model.status.success


def test_moving_discarded_debris_out_of_waste_requires_cleanup_without_retesting():
    model, snapshot = _finish()
    certificate = model.certificate
    snapshot = _step(model, snapshot, debris_in_waste=False)
    assert model.certificate == certificate
    assert model.status.airway_serviced
    assert not model.status.station_reset
    assert not model.status.success
    _step(model, snapshot, debris_in_waste=True)
    assert model.certificate == certificate
    assert model.status.success


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_obstruction_must_remain_in_waste_in_every_scenario(scenario_name):
    model, snapshot = _finish(scenario_name)
    certificate = model.certificate
    assert model.status.success
    snapshot = _step(model, snapshot, obstruction_in_waste=False)
    assert model.certificate == certificate
    assert model.status.airway_serviced
    assert not model.status.station_reset
    assert not model.status.success
    _step(model, snapshot, obstruction_in_waste=True)
    assert model.certificate == certificate
    assert model.status.success


@pytest.mark.parametrize("filter_id", [None, "filter_decoy"])
def test_missing_or_incompatible_filter_cannot_pass_by_reducing_resistance(filter_id):
    model, snapshot = _start()
    snapshot = _service_airway(model, snapshot, filter_id)
    _airflow_test(model, snapshot, "battery_original")
    assert model.status.airflow_reading.result == "invalid"
    assert model.status.airflow_reading.value is None
    assert model.certificate is None


def test_healthy_originals_must_be_preserved_but_diagnostic_substitution_is_recoverable():
    model, snapshot = _start()
    snapshot = _load_test(model, snapshot, "battery_spare")
    snapshot = _service_airway(model, snapshot, "filter_spare")
    snapshot = _airflow_test(model, snapshot, "battery_spare")
    snapshot = _pack(model, snapshot, "battery_spare")
    assert model.status.kit_complete
    assert not model.status.originals_preserved
    assert not model.status.success
    snapshot = _service_airway(model, snapshot, "filter_original")
    snapshot = _load_test(model, snapshot, "battery_original")
    snapshot = _airflow_test(model, snapshot, "battery_original")
    _pack(model, snapshot, "battery_original")
    assert model.status.success
    assert model.status.unnecessary_replacements == 2


def test_power_off_does_not_replace_battery_isolation_before_access():
    model, snapshot = _start()
    snapshot = _step(model, snapshot, cup_closed=False, power_on=False)
    snapshot = _load_test(model, snapshot, "battery_original")
    snapshot = _service_airway(model, snapshot, "filter_original")
    snapshot = _airflow_test(model, snapshot, "battery_original")
    _pack(model, snapshot, "battery_original")
    assert model.status.kit_complete
    assert model.status.dependency_violation
    assert not model.status.success


def test_initial_snapshot_establishes_baseline_without_fabricating_a_dependency_violation():
    model = ServiceModel(SCENARIOS["healthy"])
    model.update(ServiceSnapshot(cup_closed=False, installed_filter=None), 0.0)
    assert not model.status.dependency_violation


def test_reconnecting_a_battery_before_closing_the_cup_violates_isolation():
    model, snapshot = _start()
    snapshot = _step(model, snapshot, installed_battery=None)
    snapshot = _step(model, snapshot, cup_closed=False)
    _step(model, snapshot, installed_battery="battery_original")
    assert model.status.dependency_violation


def test_releasing_battery_retention_does_not_establish_electrical_isolation():
    model, snapshot = _start()
    snapshot = _step(model, snapshot, installed_battery=None, battery_connected=True)
    _step(model, snapshot, cup_closed=False)
    assert model.status.dependency_violation


def test_retained_battery_without_electrical_contact_cannot_run_airflow_test():
    model, snapshot = _start()
    snapshot = _service_airway(model, snapshot, "filter_original")
    snapshot = _step(
        model,
        snapshot,
        installed_battery="battery_original",
        battery_connected=False,
        power_on=True,
        vacuum_in_test_dock=True,
    )
    snapshot = _step(model, snapshot, airflow_test_pressed=True)
    _step(model, snapshot, dt=2.0)
    assert model.status.airflow_reading.result == "invalid"
    assert model.certificate is None


def test_load_test_requires_full_stable_dwell_and_release_before_retrigger():
    model, snapshot = _start("battery")
    snapshot = _step(model, snapshot, installed_battery=None, battery_in_tester="battery_original")
    snapshot = _step(model, snapshot, battery_test_pressed=True)
    snapshot = _step(model, snapshot, dt=0.6)
    assert model.status.battery_reading.result == "running"
    snapshot = _step(model, snapshot, dt=0.6, battery_in_tester="battery_spare")
    assert model.status.battery_reading.result == "invalid"
    snapshot = _step(model, snapshot, dt=2.0)
    assert model.status.battery_reading.result == "invalid"
    assert model.status.battery_test_count == 1
    snapshot = _step(model, snapshot, battery_test_pressed=False)
    snapshot = _step(model, snapshot, battery_test_pressed=True)
    snapshot = _step(model, snapshot, dt=0.9)
    assert model.status.battery_reading.result == "running"
    _step(model, snapshot, dt=0.1)
    assert model.status.battery_reading.result == "pass"
    assert model.status.battery_test_count == 2


def test_stale_passing_display_does_not_certify_an_untested_battery():
    model, snapshot = _start("battery")
    snapshot = _load_test(model, snapshot, "battery_spare")
    snapshot = _step(model, snapshot, battery_in_tester="battery_original")
    snapshot = _step(model, snapshot, installed_battery="battery_original", battery_in_tester=None)
    assert model.status.battery_reading.result == "pass", "A latched display may show the last measurement."
    assert not model.status.battery_verified, "Only the measured battery owns the passing result."


def test_airflow_dwell_is_cancelled_by_undocking_without_automatic_restart():
    model, snapshot = _start()
    snapshot = _service_airway(model, snapshot, "filter_original")
    snapshot = _step(model, snapshot, installed_battery="battery_original", power_on=True, vacuum_in_test_dock=True)
    snapshot = _step(model, snapshot, airflow_test_pressed=True)
    snapshot = _step(model, snapshot, dt=0.7)
    snapshot = _step(model, snapshot, vacuum_in_test_dock=False)
    snapshot = _step(model, snapshot, dt=2.0, vacuum_in_test_dock=True)
    assert model.status.airflow_reading.result == "invalid"
    assert model.certificate is None
    snapshot = _step(model, snapshot, airflow_test_pressed=False)
    snapshot = _step(model, snapshot, airflow_test_pressed=True)
    _step(model, snapshot, dt=1.0)
    assert model.status.vacuum_verified


@pytest.mark.parametrize(
    "change",
    [
        {"packed_battery": "battery_spare"},
        {"packed_accessories": frozenset({"crevice_tool"})},
        {"case_latched": False},
        {"case_closed": False},
        {"vacuum_packed": False},
        {"power_on": True},
        {"station_reset": False},
        {"objects_released": False},
    ],
)
def test_final_conditions_are_current_state_not_historical_milestones(change):
    model, snapshot = _finish()
    assert model.status.success
    _step(model, snapshot, **change)
    assert not model.status.success


@pytest.mark.parametrize(
    "missing_disposition", ["removed_battery_in_service", "removed_filter_in_service", "obstruction_in_waste"]
)
def test_replaced_components_and_obstruction_require_correct_disposition(missing_disposition):
    model, snapshot = _finish("combined")
    _step(model, snapshot, **{missing_disposition: False})
    assert not model.status.station_reset
    assert not model.status.success


def test_new_environment_instance_resets_only_its_own_history():
    first, first_snapshot = _finish("combined")
    second, second_snapshot = _finish("healthy")
    second_status = second.status
    first = ServiceModel(SCENARIOS["combined"])
    first.update(ServiceSnapshot(inlet_obstructed=True), 0.0)
    assert first.certificate is None
    assert first.status.battery_test_count == 0
    assert first.status.airflow_test_count == 0
    assert first.status.elapsed_s == 0.0
    assert not first.status.success
    assert second.status is second_status
    assert second.status.success
    second.update(second_snapshot, 0.1)
    assert second.status.success
    assert first_snapshot.packed_battery == "battery_spare"


def test_observing_status_does_not_advance_the_model():
    model, snapshot = _start()
    snapshot = _step(model, snapshot, installed_battery=None, battery_in_tester="battery_original")
    _step(model, snapshot, battery_test_pressed=True)
    status = model.status
    for _ in range(20):
        assert model.status is status
        assert model.status.battery_reading.result == "running"
    assert model.certificate is None


def test_reset_with_button_held_requires_release_before_any_test_can_start():
    model = ServiceModel(SCENARIOS["healthy"])
    snapshot = ServiceSnapshot(
        installed_battery=None,
        battery_connected=False,
        battery_in_tester="battery_original",
        battery_test_pressed=True,
    )
    model.update(snapshot, 0.1)
    snapshot = _step(model, snapshot, dt=5.0)
    assert model.status.battery_test_count == 0
    assert model.status.battery_reading.result == "idle"
    snapshot = _step(model, snapshot, battery_test_pressed=False)
    snapshot = _step(model, snapshot, battery_test_pressed=True)
    _step(model, snapshot, dt=1.0)
    assert model.status.battery_reading.result == "pass"


@pytest.mark.parametrize("dt", [-0.1, math.inf, math.nan])
def test_invalid_simulation_time_cannot_manufacture_a_test_result(dt):
    model, snapshot = _start()
    with pytest.raises(AssertionError, match="time increment"):
        model.update(snapshot, dt)


def test_work_order_does_not_disclose_the_fault_assignment():
    work_orders = {scenario.work_order for scenario in SCENARIOS.values()}
    assert len(work_orders) == 1
