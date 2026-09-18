# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hand tools, pegboard fixtures, and the YAM workcell table used by the tool-hanging tasks."""

from typing import ClassVar

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/tool_hanging/assets"


class Wrench(LibraryObject):
    """Dark 19 mm ring spanner."""

    name = "tool_hanging_wrench"
    tags = ["object", "graspable"]
    usd_path = f"{ASSET_ROOT}/vabar_tool_hanging__wrench_ring_19/vabar_tool_hanging__wrench_ring_19.usda"


class Scissors(LibraryObject):
    """Household scissors with two finger bows."""

    name = "tool_hanging_scissors"
    tags = ["object", "graspable"]
    usd_path = f"{ASSET_ROOT}/vabar_tool_hanging__scissor_pair_1/vabar_tool_hanging__scissor_pair_1.usda"


class Pliers(LibraryObject):
    """Combination pliers."""

    name = "tool_hanging_pliers"
    tags = ["object", "graspable"]
    usd_path = f"{ASSET_ROOT}/vabar_tool_hanging__plier_combination/vabar_tool_hanging__plier_combination.usda"


class Screwdriver(LibraryObject):
    """Slim flat-head screwdriver."""

    name = "tool_hanging_screwdriver"
    tags = ["object", "graspable"]
    usd_path = f"{ASSET_ROOT}/vabar_tool_hanging__screwdriver_slim/vabar_tool_hanging__screwdriver_slim.usda"


class Fixture(LibraryObject):
    """Kinematic pegboard fixture; its USD authors ``physics:kinematicEnabled``."""

    tags = ["object", "fixture"]

    def __init__(self, initial_pose: Pose | None = None, collision_mode: str = "mesh", **kwargs):
        super().__init__(initial_pose=initial_pose, collision_mode=collision_mode, **kwargs)

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Enclose the fixture under its full mount rotation; hooks are pitched as well as yawed."""
        initial_pose = self.get_initial_pose()
        if not isinstance(initial_pose, Pose):
            return super().get_world_bounding_box()
        return (
            self.get_bounding_box()
            .enclosing_after_rotation(initial_pose.rotation_xyzw)
            .translated(initial_pose.position_xyz)
        )


class PegboardRack(Fixture):
    """Four-panel pegboard standing behind the table."""

    name = "pegboard_rack"
    usd_path = f"{ASSET_ROOT}/industrial__pegboard_rack/industrial__pegboard_rack.usda"


class PegboardHook(Fixture):
    """J-hook with a convex-decomposed cavity for hanging wrenches and scissors."""

    name = "pegboard_hook"
    usd_path = f"{ASSET_ROOT}/industrial__wrench_hooks/wrench_hook_2.usda"


class PlierSupport(Fixture):
    """Triangular support that pliers straddle."""

    name = "plier_support"
    usd_path = f"{ASSET_ROOT}/industrial__plier_supports/plier_support_left.usda"


class ScrewdriverBox(Fixture):
    """Open box on the pegboard that receives an upright screwdriver."""

    name = "screwdriver_box"
    usd_path = f"{ASSET_ROOT}/industrial__screwdriver_box/box.usda"


class ScrewdriverBoxUpstream(ScrewdriverBox):
    """Screwdriver box with the wall geometry measured from AUTOLab's stored mesh."""

    name = "screwdriver_box_upstream"
    usd_path = f"{ASSET_ROOT}/industrial__screwdriver_box/box_upstream.usda"


class YamWorkcellTable(Background):
    """Stainless workcell table carrying both YAM mounts; ``placement_surface`` is its top."""

    name = "yam_workcell_table"
    tags: ClassVar[list[str]] = ["background"]
    usd_path = f"{ASSET_ROOT}/industrial__yam_workcell_table/industrial__yam_workcell_table.usda"

    def __init__(self, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            usd_path=self.usd_path,
            object_min_z=0.0,
            initial_pose=initial_pose,
            tags=self.tags,
            **kwargs,
        )
