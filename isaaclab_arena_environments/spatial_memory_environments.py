# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered environments of the spatial-memory task suite; the tasks live in ``spatial_memory.tasks``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.spatial_memory.environment import SpatialMemoryEnvironmentCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

_TASKS_PACKAGE = "isaaclab_arena_environments.spatial_memory.tasks"


class _SpatialMemoryEnvironmentBuilder:
    """Shared ``build`` of the suite's environment factories."""

    name: str
    task_logic_path: str
    """Dotted path of the task's ``SpatialMemoryTaskLogic`` subclass, imported when the environment is built."""

    def build(self, cfg: SpatialMemoryEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.utils.class_path import import_class_from_path
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment

        return build_spatial_memory_environment(self.name, import_class_from_path(self.task_logic_path), cfg)


@dataclass
class SpatialMemoryMugPastDoorEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_mug_past_door``; ``difficulty`` is the door opening angle in degrees."""


@register_environment
class SpatialMemoryMugPastDoorEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryMugPastDoorEnvironmentCfg]
):
    """Put the mug on a cabinet shelf past the open door (Franka, wrist camera)."""

    name = "spatial_memory_mug_past_door"
    _legacy_argparse_cfg_type = SpatialMemoryMugPastDoorEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.mug_past_door.MugPastDoorLogic"


@dataclass
class SpatialMemoryFetchBehindEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_fetch_behind``; ``difficulty`` is the number of mugs (2 to 6)."""


@register_environment
class SpatialMemoryFetchBehindEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryFetchBehindEnvironmentCfg]
):
    """Fetch the instructed mug from the shelf behind the robot (Franka, wrist camera)."""

    name = "spatial_memory_fetch_behind"
    _legacy_argparse_cfg_type = SpatialMemoryFetchBehindEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.fetch_behind.FetchBehindLogic"


@dataclass
class SpatialMemoryBoxCarryEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_box_carry``; ``difficulty`` is the box height in metres."""


@register_environment
class SpatialMemoryBoxCarryEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryBoxCarryEnvironmentCfg]
):
    """Carry a box past a chair and a counter corner (G1 humanoid, head camera)."""

    name = "spatial_memory_box_carry"
    _legacy_argparse_cfg_type = SpatialMemoryBoxCarryEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.box_carry.BoxCarryLogic"


@dataclass
class SpatialMemoryDishwasherRackEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_dishwasher_rack``; ``difficulty`` is the slot clearance in metres."""


@register_environment
class SpatialMemoryDishwasherRackEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryDishwasherRackEnvironmentCfg]
):
    """Load an item into a slot of the lower dishwasher rack (Franka, wrist camera)."""

    name = "spatial_memory_dishwasher_rack"
    _legacy_argparse_cfg_type = SpatialMemoryDishwasherRackEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.dishwasher_rack.DishwasherRackLogic"


@dataclass
class SpatialMemoryBackOfShelfEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_back_of_shelf``; ``difficulty`` is the gap width between the front items in metres."""


@register_environment
class SpatialMemoryBackOfShelfEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryBackOfShelfEnvironmentCfg]
):
    """Take the item from the back of the shelf (Franka, wrist camera)."""

    name = "spatial_memory_back_of_shelf"
    _legacy_argparse_cfg_type = SpatialMemoryBackOfShelfEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.back_of_shelf.BackOfShelfLogic"


@dataclass
class SpatialMemoryHighShelfEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_high_shelf``; ``difficulty`` is the number of items already on the shelf (1 to 4)."""


@register_environment
class SpatialMemoryHighShelfEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryHighShelfEnvironmentCfg]
):
    """Put an item into a free spot on a high shelf (Franka, wrist camera)."""

    name = "spatial_memory_high_shelf"
    _legacy_argparse_cfg_type = SpatialMemoryHighShelfEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.high_shelf.HighShelfLogic"


@dataclass
class SpatialMemoryMachineTendingEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_machine_tending``; ``difficulty`` is the opening width in metres."""


@register_environment
class SpatialMemoryMachineTendingEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryMachineTendingEnvironmentCfg]
):
    """Seat a part in a machine through its front opening (Franka, wrist camera)."""

    name = "spatial_memory_machine_tending"
    _legacy_argparse_cfg_type = SpatialMemoryMachineTendingEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.machine_tending.MachineTendingLogic"


@dataclass
class SpatialMemoryKittingEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_kitting``; ``difficulty`` is the number of bins (2 to 6)."""


@register_environment
class SpatialMemoryKittingEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryKittingEnvironmentCfg]
):
    """Kit the requested items from a bin rack into a tray (Franka, wrist camera)."""

    name = "spatial_memory_kitting"
    _legacy_argparse_cfg_type = SpatialMemoryKittingEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.kitting.KittingLogic"


@dataclass
class SpatialMemoryRestockSlotEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_restock_slot``; ``difficulty`` is the number of empty bins (1 to 3)."""


@register_environment
class SpatialMemoryRestockSlotEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryRestockSlotEnvironmentCfg]
):
    """Restock a new item into an empty bin (Franka, wrist camera)."""

    name = "spatial_memory_restock_slot"
    _legacy_argparse_cfg_type = SpatialMemoryRestockSlotEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.restock_slot.RestockSlotLogic"


@dataclass
class SpatialMemoryMovedCartEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure ``spatial_memory_moved_cart``; ``difficulty`` is the re-observation window in seconds."""


@register_environment
class SpatialMemoryMovedCartEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryMovedCartEnvironmentCfg]
):
    """Put the mug away after a cart was moved out of view (Franka, wrist camera)."""

    name = "spatial_memory_moved_cart"
    _legacy_argparse_cfg_type = SpatialMemoryMovedCartEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.moved_cart.MovedCartLogic"


@dataclass
class SpatialMemoryReverseGapEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 reverse-gap delivery task."""


@register_environment
class SpatialMemoryReverseGapEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryReverseGapEnvironmentCfg]
):
    """Carry a box backwards through a previously inspected barrier."""

    name = "spatial_memory_reverse_gap"
    _legacy_argparse_cfg_type = SpatialMemoryReverseGapEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.reverse_gap.ReverseGapLogic"


@dataclass
class SpatialMemoryShelfBayEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 shelf restocking task."""


@register_environment
class SpatialMemoryShelfBayEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryShelfBayEnvironmentCfg]
):
    """Restock a previously inspected shelf bay."""

    name = "spatial_memory_shelf_bay"
    _legacy_argparse_cfg_type = SpatialMemoryShelfBayEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.shelf_bay.ShelfBayLogic"


@dataclass
class SpatialMemoryPartitionDeliveryEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 partition delivery task."""


@register_environment
class SpatialMemoryPartitionDeliveryEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryPartitionDeliveryEnvironmentCfg]
):
    """Deliver a carton to the room matching the delayed instruction."""

    name = "spatial_memory_partition_delivery"
    _legacy_argparse_cfg_type = SpatialMemoryPartitionDeliveryEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.partition_delivery.PartitionDeliveryLogic"
