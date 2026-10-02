# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functional instruments and completion rules driven by physical scene snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .scenarios import ServiceScenario


@dataclass(frozen=True)
class ServiceSnapshot:
    """Describe measured scene state for one environment at one control step."""

    installed_battery: str | None = "battery_original"
    """Identity of the battery physically seated and secured in the vacuum."""

    battery_connected: bool = True
    """Whether a battery still occupies the electrical contacts, even after releasing retention."""

    installed_filter: str | None = "filter_original"
    """Identity of the cartridge seated with the correct position and orientation."""

    cup_closed: bool = True
    """Whether the collection cup is seated and its retaining latch is closed."""

    inlet_obstructed: bool = False
    """Whether the physical obstruction occupies the inlet passage."""

    cup_debris_present: bool = True
    """Whether any loose debris still occupies the collection cup."""

    debris_in_waste: bool = False
    """Whether all cup debris is contained in the waste tray."""

    battery_in_tester: str | None = None
    """Identity of the battery correctly seated in the load-test fixture."""

    battery_test_pressed: bool = False
    """Whether the load-test button is physically depressed."""

    vacuum_in_test_dock: bool = False
    """Whether the assembled vacuum is correctly seated in the airflow fixture."""

    airflow_test_pressed: bool = False
    """Whether the airflow-test button is physically depressed."""

    power_on: bool = False
    """Whether the vacuum is energized by its switch or the interlocked test dock."""

    vacuum_packed: bool = False
    """Whether the vacuum is correctly seated in its carrying-case compartment."""

    packed_battery: str | None = None
    """Identity of the battery in the separate carrying-case battery compartment."""

    packed_accessories: frozenset[str] = frozenset()
    """Identities of accessories correctly seated in the case."""

    case_closed: bool = False
    """Whether the carrying-case lid is fully closed."""

    case_latched: bool = False
    """Whether every carrying-case latch is engaged."""

    removed_battery_in_service: bool = False
    """Whether the original battery is contained in the battery-service tray."""

    removed_filter_in_service: bool = False
    """Whether the original filter is contained in the filter-service tray."""

    obstruction_in_waste: bool = False
    """Whether the removed obstruction is contained in the waste tray."""

    station_reset: bool = False
    """Whether unused parts, tools, and fixtures are in their designated ready positions."""

    objects_released: bool = False
    """Whether the gripper has released all task objects and the final objects have settled."""


@dataclass(frozen=True)
class InstrumentReading:
    """Expose only the measurement and operating state available on an instrument."""

    result: Literal["idle", "running", "pass", "fail", "invalid"] = "idle"
    """Current displayed state; invalid means a seating or assembly interlock failed."""

    value: float | None = None
    """Measured voltage or airflow; absent before measurement and after interruption."""


@dataclass(frozen=True)
class TestCertificate:
    """Associate a passing assembled-vacuum test with the exact serviced assembly."""

    battery_id: str
    """Identity of the battery used during the passing test."""

    filter_id: str
    """Identity of the cartridge used during the passing test."""

    airway_revision: int
    """Monotonic episode-local revision changed whenever the airflow assembly changes."""


@dataclass(frozen=True)
class ServiceStatus:
    """Report current completion conditions and evaluator metrics without advancing time."""

    success: bool = False
    """Whether every current completion condition is satisfied."""

    battery_verified: bool = False
    """Whether the installed or packed battery owns a passing load-test result."""

    airway_serviced: bool = False
    """Whether the cup is empty and closed and the airflow path is healthy."""

    vacuum_verified: bool = False
    """Whether a passing airflow certificate remains valid for the current assembly."""

    originals_preserved: bool = False
    """Whether each healthy original component is selected for the final kit."""

    kit_complete: bool = False
    """Whether the powered-off, tested assembly and requested contents are secured in the case."""

    station_reset: bool = False
    """Whether parts are correctly routed, the station is restored, and objects are released."""

    dependency_violation: bool = False
    """Whether this episode ever accessed the airflow assembly without isolating the battery."""

    battery_reading: InstrumentReading = InstrumentReading()
    """Load-tester display, in volts."""

    airflow_reading: InstrumentReading = InstrumentReading()
    """Airflow-tester display, in liters per second."""

    battery_test_count: int = 0
    """Number of load-test attempts, including interrupted and interlocked attempts."""

    airflow_test_count: int = 0
    """Number of airflow-test attempts, including interrupted and interlocked attempts."""

    unnecessary_replacements: int = 0
    """Number of non-original insertions when the corresponding original was healthy."""

    elapsed_s: float = 0.0
    """Elapsed simulation time since this episode's model was constructed."""


@dataclass
class _PendingTest:
    key: tuple
    elapsed_s: float = 0.0


class ServiceModel:
    """Advance one episode from physical evidence, allowing diagnosis and rework in any valid order."""

    NOMINAL_AIRFLOW_LPS = 22.0
    MIN_AIRFLOW_LPS = 18.0
    MIN_BATTERY_VOLTAGE = 18.0

    def __init__(self, scenario: ServiceScenario, test_duration_s: float = 1.0) -> None:
        assert math.isfinite(test_duration_s) and test_duration_s > 0, "Test duration must be finite and positive."
        self.scenario = scenario
        self.test_duration_s = test_duration_s
        self.status = ServiceStatus()
        self.certificate: TestCertificate | None = None
        self._previous: ServiceSnapshot | None = None
        self._airway_revision = 0
        self._dependency_violation = False
        self._battery_reading = InstrumentReading()
        self._airflow_reading = InstrumentReading()
        self._battery_test: _PendingTest | None = None
        self._airflow_test: _PendingTest | None = None
        self._verified_batteries: set[str] = set()
        self._battery_test_count = 0
        self._airflow_test_count = 0
        self._unnecessary_replacements = 0
        self._elapsed_s = 0.0

    def update(self, snapshot: ServiceSnapshot, dt: float) -> ServiceStatus:
        """Advance exactly once per control step and return the current status.

        Args:
            snapshot: Physical evidence for this environment; identities must come from seating checks.
            dt: Elapsed simulation seconds, finite and nonnegative. Zero is useful for initialization.

        Returns:
            An immutable status. Reading it repeatedly does not advance a pending test.
        """
        assert math.isfinite(dt) and dt >= 0, "Simulation time increment must be finite and nonnegative."
        self._elapsed_s += dt
        self._track_assembly_changes(snapshot)
        previous = self._previous
        battery_pressed = previous is not None and snapshot.battery_test_pressed and not previous.battery_test_pressed
        airflow_pressed = previous is not None and snapshot.airflow_test_pressed and not previous.airflow_test_pressed
        self._update_battery_test(snapshot, battery_pressed, dt)
        self._update_airflow_test(snapshot, airflow_pressed, dt)
        self.status = self._current_status(snapshot)
        self._previous = snapshot
        return self.status

    @staticmethod
    def _airway_state(snapshot: ServiceSnapshot) -> tuple:
        return (
            snapshot.cup_closed,
            snapshot.installed_filter,
            snapshot.inlet_obstructed,
            snapshot.cup_debris_present,
        )

    def _track_assembly_changes(self, snapshot: ServiceSnapshot) -> None:
        previous = self._previous
        if previous is None:
            return
        if self._airway_state(previous) != self._airway_state(snapshot):
            self._airway_revision += 1
            self.certificate = None
        accessed_airway = (previous.cup_closed and not snapshot.cup_closed) or (
            previous.installed_filter != snapshot.installed_filter
        )
        battery_present_during_access = accessed_airway and (previous.battery_connected or snapshot.battery_connected)
        powered_open_assembly = snapshot.battery_connected and (
            not snapshot.cup_closed or snapshot.installed_filter is None
        )
        if battery_present_during_access or powered_open_assembly:
            self._dependency_violation = True
        if self.certificate is not None and snapshot.installed_battery not in (None, self.certificate.battery_id):
            self.certificate = None
        battery_changed = previous.installed_battery != snapshot.installed_battery
        filter_changed = previous.installed_filter != snapshot.installed_filter
        if battery_changed and not self.scenario.weak_battery:
            if snapshot.installed_battery not in (None, self.scenario.original_battery):
                self._unnecessary_replacements += 1
        if filter_changed and not self.scenario.clogged_filter:
            if snapshot.installed_filter not in (None, self.scenario.original_filter):
                self._unnecessary_replacements += 1

    def _update_battery_test(self, snapshot: ServiceSnapshot, pressed: bool, dt: float) -> None:
        battery_id = snapshot.battery_in_tester
        voltage = self.scenario.battery_voltage(battery_id) if battery_id is not None else None
        key = (battery_id,) if voltage is not None and battery_id != snapshot.installed_battery else None
        if self._battery_test is not None and self._battery_test.key != key:
            self._battery_test = None
            self._battery_reading = InstrumentReading("invalid")
        if pressed:
            self._battery_test_count += 1
            self._battery_test = _PendingTest(key) if key is not None else None
            self._battery_reading = InstrumentReading("running" if key is not None else "invalid")
            return
        if self._battery_test is None:
            return
        self._battery_test.elapsed_s += dt
        if self._battery_test.elapsed_s + 1e-9 < self.test_duration_s:
            return
        passed = self.scenario.battery_compatible(battery_id) and voltage >= self.MIN_BATTERY_VOLTAGE
        self._battery_reading = InstrumentReading("pass" if passed else "fail", voltage)
        if passed:
            self._verified_batteries.add(battery_id)
        else:
            self._verified_batteries.discard(battery_id)
        self._battery_test = None

    def _airflow_test_key(self, snapshot: ServiceSnapshot) -> tuple | None:
        assembled = snapshot.cup_closed and self.scenario.filter_compatible(snapshot.installed_filter)
        can_operate = (
            snapshot.power_on
            and snapshot.battery_connected
            and self.scenario.battery_compatible(snapshot.installed_battery)
        )
        if not (snapshot.vacuum_in_test_dock and assembled and can_operate):
            return None
        return (snapshot.installed_battery, snapshot.installed_filter, self._airway_revision)

    def _update_airflow_test(self, snapshot: ServiceSnapshot, pressed: bool, dt: float) -> None:
        key = self._airflow_test_key(snapshot)
        if self._airflow_test is not None and self._airflow_test.key != key:
            self._airflow_test = None
            self._airflow_reading = InstrumentReading("invalid")
        if pressed:
            self._airflow_test_count += 1
            self._airflow_test = _PendingTest(key) if key is not None else None
            self._airflow_reading = InstrumentReading("running" if key is not None else "invalid")
            return
        if self._airflow_test is None:
            return
        self._airflow_test.elapsed_s += dt
        if self._airflow_test.elapsed_s + 1e-9 < self.test_duration_s:
            return
        airflow = self._measure_airflow(snapshot)
        passed = airflow >= self.MIN_AIRFLOW_LPS
        self._airflow_reading = InstrumentReading("pass" if passed else "fail", airflow)
        if passed:
            self.certificate = TestCertificate(*key)
        else:
            self.certificate = None
        self._airflow_test = None

    def _measure_airflow(self, snapshot: ServiceSnapshot) -> float:
        voltage = self.scenario.battery_voltage(snapshot.installed_battery)
        if voltage is None or voltage < self.MIN_BATTERY_VOLTAGE:
            return 0.0
        airflow = self.NOMINAL_AIRFLOW_LPS
        if snapshot.installed_filter == self.scenario.original_filter and self.scenario.clogged_filter:
            airflow *= 0.35
        if snapshot.inlet_obstructed:
            airflow *= 0.45
        if snapshot.cup_debris_present:
            airflow *= 0.8
        return airflow

    def _current_status(self, snapshot: ServiceSnapshot) -> ServiceStatus:
        scenario = self.scenario
        certificate = self.certificate
        vacuum_verified = certificate is not None and certificate.airway_revision == self._airway_revision
        selected_battery = snapshot.installed_battery or snapshot.packed_battery
        battery_verified = selected_battery is not None and selected_battery in self._verified_batteries
        filter_healthy = scenario.filter_compatible(snapshot.installed_filter) and not (
            scenario.clogged_filter and snapshot.installed_filter == scenario.original_filter
        )
        airway_serviced = (
            snapshot.cup_closed and filter_healthy and not snapshot.inlet_obstructed and not snapshot.cup_debris_present
        )
        originals_preserved = (scenario.weak_battery or selected_battery == scenario.original_battery) and (
            scenario.clogged_filter or snapshot.installed_filter == scenario.original_filter
        )
        kit_complete = (
            vacuum_verified
            and snapshot.vacuum_packed
            and snapshot.installed_battery is None
            and not snapshot.battery_connected
            and snapshot.packed_battery == certificate.battery_id
            and snapshot.packed_accessories == scenario.requested_accessories
            and snapshot.case_closed
            and snapshot.case_latched
            and not snapshot.power_on
        )
        disposition_complete = (
            snapshot.debris_in_waste
            and (not scenario.weak_battery or snapshot.removed_battery_in_service)
            and (not scenario.clogged_filter or snapshot.removed_filter_in_service)
            and snapshot.obstruction_in_waste
        )
        station_reset = snapshot.station_reset and snapshot.objects_released and disposition_complete
        success = (
            battery_verified
            and airway_serviced
            and originals_preserved
            and kit_complete
            and station_reset
            and not self._dependency_violation
        )
        return ServiceStatus(
            success=success,
            battery_verified=battery_verified,
            airway_serviced=airway_serviced,
            vacuum_verified=vacuum_verified,
            originals_preserved=originals_preserved,
            kit_complete=kit_complete,
            station_reset=station_reset,
            dependency_violation=self._dependency_violation,
            battery_reading=self._battery_reading,
            airflow_reading=self._airflow_reading,
            battery_test_count=self._battery_test_count,
            airflow_test_count=self._airflow_test_count,
            unnecessary_replacements=self._unnecessary_replacements,
            elapsed_s=self._elapsed_s,
        )
