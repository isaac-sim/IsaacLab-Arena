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


@dataclass
class SpatialMemoryPartitionSilhouettesEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the experimental room delivery with depth-readable plaques."""


@register_environment
class SpatialMemoryPartitionSilhouettesEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryPartitionSilhouettesEnvironmentCfg]
):
    """Carry to a requested room identifiable from the silhouette of its marker."""

    name = "spatial_memory_partition_silhouettes"
    _legacy_argparse_cfg_type = SpatialMemoryPartitionSilhouettesEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.partition_delivery.PartitionSilhouetteDeliveryLogic"


@dataclass
class SpatialMemoryDoglegExitEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 dogleg exit task."""


@register_environment
class SpatialMemoryDoglegExitEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryDoglegExitEnvironmentCfg]
):
    """Back through an inspected dogleg and deliver the carton."""

    name = "spatial_memory_dogleg_exit"
    _legacy_argparse_cfg_type = SpatialMemoryDoglegExitEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.dogleg_exit.DoglegExitLogic"


@dataclass
class SpatialMemoryTurningPocketEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 turning pocket task."""


@register_environment
class SpatialMemoryTurningPocketEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryTurningPocketEnvironmentCfg]
):
    """Turn a loaded carton around inside a previously inspected recess."""

    name = "spatial_memory_turning_pocket"
    _legacy_argparse_cfg_type = SpatialMemoryTurningPocketEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.turning_pocket.TurningPocketLogic"


@dataclass
class SpatialMemoryTwoViewExitEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 two view exit task."""


@register_environment
class SpatialMemoryTwoViewExitEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryTwoViewExitEnvironmentCfg]
):
    """Find the common clear lane through two gates."""

    name = "spatial_memory_two_view_exit"
    _legacy_argparse_cfg_type = SpatialMemoryTwoViewExitEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.two_view_exit.TwoViewExitLogic"


@dataclass
class SpatialMemoryFunnelRetreatEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 funnel retreat task."""


@register_environment
class SpatialMemoryFunnelRetreatEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryFunnelRetreatEnvironmentCfg]
):
    """Back a carton through a previously surveyed funnel."""

    name = "spatial_memory_funnel_retreat"
    _legacy_argparse_cfg_type = SpatialMemoryFunnelRetreatEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.funnel_retreat.FunnelRetreatLogic"


@dataclass
class SpatialMemoryStaggeredLipsEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 staggered lips task."""


@register_environment
class SpatialMemoryStaggeredLipsEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryStaggeredLipsEnvironmentCfg]
):
    """Withdraw a carried carton past two projecting shelf lips."""

    name = "spatial_memory_staggered_lips"
    _legacy_argparse_cfg_type = SpatialMemoryStaggeredLipsEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.staggered_lips.StaggeredLipsLogic"


@dataclass
class SpatialMemoryCurvedChannelEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 curved channel task."""


@register_environment
class SpatialMemoryCurvedChannelEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryCurvedChannelEnvironmentCfg]
):
    """Back along the remembered curvature of two rack rails."""

    name = "spatial_memory_curved_channel"
    _legacy_argparse_cfg_type = SpatialMemoryCurvedChannelEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.curved_channel.CurvedChannelLogic"


@dataclass
class SpatialMemoryPostTurnEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 post turn task."""


@register_environment
class SpatialMemoryPostTurnEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryPostTurnEnvironmentCfg]
):
    """Turn a carried carton between a remembered post and stop."""

    name = "spatial_memory_post_turn"
    _legacy_argparse_cfg_type = SpatialMemoryPostTurnEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.post_turn.PostTurnLogic"


@dataclass
class SpatialMemoryColumnGridEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the G1 column grid task."""


@register_environment
class SpatialMemoryColumnGridEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryColumnGridEnvironmentCfg]
):
    """Carry through a surveyed column grid to a dock requested at pickup."""

    name = "spatial_memory_column_grid"
    _legacy_argparse_cfg_type = SpatialMemoryColumnGridEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.column_grid.ColumnGridLogic"


@dataclass
class SpatialMemoryServiceTrenchEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the service trench task."""


@register_environment
class SpatialMemoryServiceTrenchEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryServiceTrenchEnvironmentCfg]
):
    """Walk headfirst to the end of the narrow service trench. Back out to the mouth facing the end."""

    name = "spatial_memory_service_trench"
    _legacy_argparse_cfg_type = SpatialMemoryServiceTrenchEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.ServiceTrench"


@dataclass
class SpatialMemoryDockEdgeEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the dock edge task."""


@register_environment
class SpatialMemoryDockEdgeEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryDockEdgeEnvironmentCfg]
):
    """Back toward an inspected dock sill, stop at the standoff line facing away, then walk clear."""

    name = "spatial_memory_dock_edge"
    _legacy_argparse_cfg_type = SpatialMemoryDockEdgeEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.DockEdge"


@dataclass
class SpatialMemorySidewaysStacksEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the sideways stacks task."""


@register_environment
class SpatialMemorySidewaysStacksEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemorySidewaysStacksEnvironmentCfg]
):
    """Face the shelving and sidestep between the stacks, maintaining that heading at three stations."""

    name = "spatial_memory_sideways_stacks"
    _legacy_argparse_cfg_type = SpatialMemorySidewaysStacksEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.SidewaysStacks"


@dataclass
class SpatialMemoryLowHeaderEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the low header task."""


@register_environment
class SpatialMemoryLowHeaderEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryLowHeaderEnvironmentCfg]
):
    """Lower the pelvis beneath the inspected lintel and emerge on its far side."""

    name = "spatial_memory_low_header"
    _legacy_argparse_cfg_type = SpatialMemoryLowHeaderEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.LowHeader"


@dataclass
class SpatialMemoryTankPerimeterEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the tank perimeter task."""


@register_environment
class SpatialMemoryTankPerimeterEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryTankPerimeterEnvironmentCfg]
):
    """Inspect all four faces of an enclosed tank in order and return to the first face."""

    name = "spatial_memory_tank_perimeter"
    _legacy_argparse_cfg_type = SpatialMemoryTankPerimeterEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.TankPerimeter"


@dataclass
class SpatialMemorySightlineEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the sightline task."""


@register_environment
class SpatialMemorySightlineEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemorySightlineEnvironmentCfg]
):
    """Inspect a pair of survey posts, then occupy their extended sightline and face along it."""

    name = "spatial_memory_sightline"
    _legacy_argparse_cfg_type = SpatialMemorySightlineEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.inspection_walks.Sightline"


@dataclass
class SpatialMemoryRecessedResetEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the recessed reset task."""


@register_environment
class SpatialMemoryRecessedResetEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryRecessedResetEnvironmentCfg]
):
    """Reach the left palm into the hooded reset recess, hold at the pad, then withdraw the hand."""

    name = "spatial_memory_recessed_reset"
    _legacy_argparse_cfg_type = SpatialMemoryRecessedResetEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.RecessedReset"


@dataclass
class SpatialMemoryTwoHandInterlockEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the two hand interlock task."""


@register_environment
class SpatialMemoryTwoHandInterlockEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryTwoHandInterlockEnvironmentCfg]
):
    """Hold both palms at the two separated interlock pads simultaneously."""

    name = "spatial_memory_two_hand_interlock"
    _legacy_argparse_cfg_type = SpatialMemoryTwoHandInterlockEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.TwoHandInterlock"


@dataclass
class SpatialMemorySeamTraceEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the seam trace task."""


@register_environment
class SpatialMemorySeamTraceEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemorySeamTraceEnvironmentCfg]
):
    """Follow a stepped seam with the right palm, passing the start, bend and raised end in order."""

    name = "spatial_memory_seam_trace"
    _legacy_argparse_cfg_type = SpatialMemorySeamTraceEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.SeamTrace"


@dataclass
class SpatialMemoryBlindReturnHookEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the blind return hook task."""


@register_environment
class SpatialMemoryBlindReturnHookEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryBlindReturnHookEnvironmentCfg]
):
    """Guide a palm around the free edge of a screen, behind its flange and back out."""

    name = "spatial_memory_blind_return_hook"
    _legacy_argparse_cfg_type = SpatialMemoryBlindReturnHookEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.BlindReturnHook"


@dataclass
class SpatialMemoryFloorSledEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the floor sled task."""


@register_environment
class SpatialMemoryFloorSledEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryFloorSledEnvironmentCfg]
):
    """Move the low floor sled down its guide to a remembered bumper, then clear the sled."""

    name = "spatial_memory_floor_sled"
    _legacy_argparse_cfg_type = SpatialMemoryFloorSledEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.FloorSled"


@dataclass
class SpatialMemoryCounterSlideEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the counter slide task."""


@register_environment
class SpatialMemoryCounterSlideEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryCounterSlideEnvironmentCfg]
):
    """Slide a tray laterally under a low pass-through cover and leave it at the outlet."""

    name = "spatial_memory_counter_slide"
    _legacy_argparse_cfg_type = SpatialMemoryCounterSlideEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.CounterSlide"


@dataclass
class SpatialMemoryJigRotationEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the jig rotation task."""


@register_environment
class SpatialMemoryJigRotationEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryJigRotationEnvironmentCfg]
):
    """Quarter-turn a long workpiece on the bench and settle it beside the locating fence."""

    name = "spatial_memory_jig_rotation"
    _legacy_argparse_cfg_type = SpatialMemoryJigRotationEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.JigRotation"


@dataclass
class SpatialMemoryToteExtractionEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the tote extraction task."""


@register_environment
class SpatialMemoryToteExtractionEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryToteExtractionEnvironmentCfg]
):
    """Draw a tote out from under a shelf lip. Bring it beyond the apron before raising it."""

    name = "spatial_memory_tote_extraction"
    _legacy_argparse_cfg_type = SpatialMemoryToteExtractionEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.ToteExtraction"


@dataclass
class SpatialMemoryScaleTransferEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the scale transfer task."""


@register_environment
class SpatialMemoryScaleTransferEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryScaleTransferEnvironmentCfg]
):
    """Set the case down on the scale, release it for a reading, then remove it to the outbound pad."""

    name = "spatial_memory_scale_transfer"
    _legacy_argparse_cfg_type = SpatialMemoryScaleTransferEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.ScaleTransfer"


@dataclass
class SpatialMemoryRaisedSillEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the raised sill task."""


@register_environment
class SpatialMemoryRaisedSillEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryRaisedSillEnvironmentCfg]
):
    """Lift the case above a raised sill, carry it across the divider and lower it onto the far support."""

    name = "spatial_memory_raised_sill"
    _legacy_argparse_cfg_type = SpatialMemoryRaisedSillEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.manual_work.RaisedSill"


@dataclass
class SpatialMemoryCartRefugeEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the cart refuge task."""


@register_environment
class SpatialMemoryCartRefugeEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryCartRefugeEnvironmentCfg]
):
    """Yield inside an inspected passing recess as the cart goes by; then rejoin the aisle."""

    name = "spatial_memory_cart_refuge"
    _legacy_argparse_cfg_type = SpatialMemoryCartRefugeEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.CartRefuge"


@dataclass
class SpatialMemorySwingingBoomEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the swinging boom task."""


@register_environment
class SpatialMemorySwingingBoomEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemorySwingingBoomEnvironmentCfg]
):
    """Round the outside of a sweeping boom and reach the service side of its pedestal."""

    name = "spatial_memory_swinging_boom"
    _legacy_argparse_cfg_type = SpatialMemorySwingingBoomEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.SwingingBoom"


@dataclass
class SpatialMemoryLiftGateEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the lift gate task."""


@register_environment
class SpatialMemoryLiftGateEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryLiftGateEnvironmentCfg]
):
    """Stage at the remembered portal, cross beneath the raised shutter and clear its exit."""

    name = "spatial_memory_lift_gate"
    _legacy_argparse_cfg_type = SpatialMemoryLiftGateEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.LiftGate"


@dataclass
class SpatialMemoryShuttleEscortEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the shuttle escort task."""


@register_environment
class SpatialMemoryShuttleEscortEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryShuttleEscortEnvironmentCfg]
):
    """Meet the shuttle beside its guide rail, escort it past the midpoint and leave at the far berth."""

    name = "spatial_memory_shuttle_escort"
    _legacy_argparse_cfg_type = SpatialMemoryShuttleEscortEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.ShuttleEscort"


@dataclass
class SpatialMemoryMovingReceiverEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the moving receiver task."""


@register_environment
class SpatialMemoryMovingReceiverEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryMovingReceiverEnvironmentCfg]
):
    """Lift the parcel from the fixed bench and release it on the moving receiver near the dispatch end."""

    name = "spatial_memory_moving_receiver"
    _legacy_argparse_cfg_type = SpatialMemoryMovingReceiverEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.MovingReceiver"


@dataclass
class SpatialMemoryBollardChicaneEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure the bollard chicane task."""


@register_environment
class SpatialMemoryBollardChicaneEnvironment(
    _SpatialMemoryEnvironmentBuilder, ArenaEnvironmentFactory[SpatialMemoryBollardChicaneEnvironmentCfg]
):
    """Negotiate two offset retracting bollards. Cross each footprint only when that bollard is down."""

    name = "spatial_memory_bollard_chicane"
    _legacy_argparse_cfg_type = SpatialMemoryBollardChicaneEnvironmentCfg
    task_logic_path = f"{_TASKS_PACKAGE}.traffic_work.BollardChicane"
