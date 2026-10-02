# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode definitions for the returned-vacuum service task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceScenario:
    """Specify evaluator-only component conditions and the public work order."""

    name: str
    """Stable evaluation variation name; exclude it from policy observations."""

    weak_battery: bool = False
    """Whether the original precharged battery fails a load test."""

    clogged_filter: bool = False
    """Whether the original filter requires exchange for a clean cartridge."""

    inlet_obstruction: bool = False
    """Whether the inlet initially contains a removable rigid obstruction."""

    requested_accessories: frozenset[str] = frozenset({"crevice_tool", "brush_tool"})
    """Accessories named in the public work order."""

    original_battery: str = "battery_original"
    """Identity of the battery returned with the vacuum."""

    spare_battery: str = "battery_spare"
    """Identity of the compatible, healthy precharged replacement battery."""

    incompatible_battery: str = "battery_decoy"
    """Identity of the visibly incompatible battery supplied as a distractor."""

    original_filter: str = "filter_original"
    """Identity of the filter returned with the vacuum."""

    spare_filter: str = "filter_spare"
    """Identity of the compatible clean filter cartridge."""

    incompatible_filter: str = "filter_decoy"
    """Identity of the visibly incompatible filter supplied as a distractor."""

    def __post_init__(self) -> None:
        assert self.name, "Each scenario needs a name."
        battery_ids = (self.original_battery, self.spare_battery, self.incompatible_battery)
        filter_ids = (self.original_filter, self.spare_filter, self.incompatible_filter)
        component_ids = battery_ids + filter_ids
        assert all(component_ids), "Component identities must be nonempty."
        assert len(set(component_ids)) == len(component_ids), "Component identities must be unique."
        assert self.requested_accessories <= {"crevice_tool", "brush_tool"}, "Unknown accessory in work order."

    @property
    def work_order(self) -> str:
        """Return instructions without revealing the fault assignment."""
        accessory_labels = {"brush_tool": "brush tool", "crevice_tool": "crevice tool"}
        accessories = " and ".join(accessory_labels[name] for name in sorted(self.requested_accessories))
        accessories = f"the {accessories}" if accessories else "no accessories"
        return (
            "Return this handheld vacuum to service. All supplied batteries are precharged. "
            "Empty the collection cup, diagnose and resolve battery "
            "or airflow problems, and preserve working original components. Disconnect the battery before "
            "opening the cup or removing the filter. Load-test the battery and verify the assembled vacuum "
            "in the airflow dock. Pack the vacuum and the same tested battery in separate compartments "
            f"of the carrying case. Include {accessories}. Close and latch the case. Route removed batteries and "
            "filters to their service trays, discard debris and the removed obstruction, and restore the station."
        )

    def battery_voltage(self, battery_id: str | None) -> float | None:
        """Return simulated volts under load, or None for an unknown battery."""
        if battery_id == self.original_battery:
            return 13.5 if self.weak_battery else 20.0
        if battery_id == self.spare_battery:
            return 20.0
        if battery_id == self.incompatible_battery:
            return 12.0
        return None

    def battery_compatible(self, battery_id: str | None) -> bool:
        """Return whether the identified battery fits the vacuum's electrical interface."""
        return battery_id in (self.original_battery, self.spare_battery)

    def filter_compatible(self, filter_id: str | None) -> bool:
        """Return whether the identified cartridge fits the vacuum's filter interface."""
        return filter_id in (self.original_filter, self.spare_filter)


SCENARIOS = {
    "healthy": ServiceScenario("healthy"),
    "battery": ServiceScenario("battery", weak_battery=True),
    "filter": ServiceScenario("filter", clogged_filter=True),
    "obstruction": ServiceScenario("obstruction", inlet_obstruction=True),
    "battery_filter": ServiceScenario("battery_filter", weak_battery=True, clogged_filter=True),
    "battery_obstruction": ServiceScenario("battery_obstruction", weak_battery=True, inlet_obstruction=True),
    "filter_obstruction": ServiceScenario("filter_obstruction", clogged_filter=True, inlet_obstruction=True),
    "combined": ServiceScenario("combined", weak_battery=True, clogged_filter=True, inlet_obstruction=True),
}
"""The full three-fault factorial, including a healthy-return control."""
