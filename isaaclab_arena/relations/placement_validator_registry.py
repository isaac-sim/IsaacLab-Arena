# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import Registry

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_validators import PlacementValidator


class PlacementValidatorRegistry(Registry):
    """Registry for PlacementValidator subclasses, keyed by the check name they report.

    Unlike the asset registries, this takes no part in the asset cascade: it self-populates when
    placement_validators is imported, always before any build_validators() lookup.
    """

    def get_validator_by_name(self, check: str) -> type[PlacementValidator]:
        """Gets a placement validator class by the check name it reports.

        Args:
            check: The placement check name whose validator class to fetch.
        """
        return self.get_component_by_name(check)


def register_validator(cls):
    """Class decorator registering a PlacementValidator subclass under its ``check`` name.

    Keyed by ``cls.check`` (the check it reports) so build_validators() can resolve it.
    """
    registry = PlacementValidatorRegistry()
    if registry.is_registered(cls.check, ensure_loaded=False):
        print(f"WARNING: Placement validator for {cls.check} is already registered. Doing nothing.")
    else:
        registry.register(cls, cls.check)
    return cls
