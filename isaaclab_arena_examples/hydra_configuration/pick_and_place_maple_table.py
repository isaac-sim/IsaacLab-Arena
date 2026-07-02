# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hydra-native Maple-table definition with a temporary legacy adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironment
from isaaclab_arena_examples.hydra_configuration.config import (
    ArenaEnvironmentConfiguration,
    register_environment_configuration,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment_configuration("pick_and_place_maple_table")
@dataclass
class PickAndPlaceMapleTableEnvironmentConfiguration(ArenaEnvironmentConfiguration):
    """Configure and build the Maple-table pick-and-place environment.

    This Hydra-native definition is not the runtime environment. It is the intended
    replacement for the legacy ``PickAndPlaceMapleTableEnvironment`` factory.
    """

    embodiment_asset_name: str = "droid_abs_joint_pos"
    teleoperation_device_name: str | None = None
    high_dynamic_range_image_name: str | None = None
    light_intensity: float = 500.0
    pick_up_object_asset_name: str = "rubiks_cube_hot3d_robolab"
    destination_location_asset_name: str = "bowl_ycb_robolab"
    additional_table_object_asset_names: list[str] = field(default_factory=list)

    def build(self, *, enable_cameras: bool) -> IsaacLabArenaEnvironment:
        """Build through the MVP's temporary legacy compatibility adapter."""
        # Once migrated, the legacy get_env() body moves here and this Namespace bridge disappears.
        legacy_environment_factory = PickAndPlaceMapleTableEnvironment()
        legacy_arguments = argparse.Namespace(
            embodiment=self.embodiment_asset_name,
            teleop_device=self.teleoperation_device_name,
            hdr=self.high_dynamic_range_image_name,
            light_intensity=self.light_intensity,
            pick_up_object=self.pick_up_object_asset_name,
            destination_location=self.destination_location_asset_name,
            additional_table_objects=self.additional_table_object_asset_names,
            enable_cameras=enable_cameras,
        )
        arena_environment = legacy_environment_factory.get_env(legacy_arguments)
        if self.teleoperation_device_name is not None:
            arena_environment.teleop_device = legacy_environment_factory.device_registry.get_device_by_name(
                self.teleoperation_device_name
            )()
        return arena_environment
