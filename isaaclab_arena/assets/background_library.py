# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


class LibraryBackground(Background):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str | None = None
    initial_pose: Pose | None = None
    object_min_z: float
    spawn_cfg_addon: dict[str, Any] = {}
    asset_cfg_addon: dict[str, Any] = {}

    def __init__(self, **kwargs):
        # Check lazy USD paths are set by here
        assert self.usd_path is not None
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            initial_pose=self.initial_pose,
            object_min_z=self.object_min_z,
            spawn_cfg_addon=self.spawn_cfg_addon,
            asset_cfg_addon=self.asset_cfg_addon,
            **kwargs,
        )


@register_asset
class KitchenBackground(LibraryBackground):
    """
    Encapsulates the background scene for the kitchen.
    """

    name = "kitchen"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/kitchen_background/kitchen_background.usd"
    initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class KitchenWithOpenDrawerBackground(LibraryBackground):
    """
    Encapsulates the background scene for the kitchen with an open drawer.
    """

    name = "kitchen_with_open_drawer"
    tags = ["background"]
    usd_path = (
        f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/kitchen_scene_teleop_v3/kitchen_scene_teleop_v3.usd"
    )
    initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class PackingTableBackground(LibraryBackground):
    """
    Encapsulates the background scene for the packing table.
    """

    name = "packing_table"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/packing_table/packing_table.usd"
    initial_pose = Pose(position_xyz=(0.72193, -0.04727, -0.92512), rotation_wxyz=(0.70711, 0.0, 0.0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class GalileoBackground(LibraryBackground):
    """
    Encapsulates the background scene for the galileo room.
    """

    name = "galileo"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/galileo_simplified/galileo_simplified.usd"
    initial_pose = Pose(position_xyz=(4.420, 1.408, -0.795), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class GalileoLocomanipBackground(LibraryBackground):
    """
    Encapsulates the background scene for the galileo room for locomanip.
    """

    name = "galileo_locomanip"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"
    initial_pose = Pose(position_xyz=(4.420, 1.408, -0.795), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class Table(LibraryBackground):
    """
    A table.
    """

    name = "table"
    tags = ["background"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    object_min_z = -0.05

    def __init__(self):
        super().__init__()


@register_asset
class LightwheelKitchenBackground(LibraryBackground):
    """
    Encapsulates the background scene for the Lightwheel Robocasa kitchen.
    """

    name = "lightwheel_robocasa_kitchen"
    tags = ["background"]
    usd_path = None
    initial_pose = Pose.identity()
    object_min_z = -0.2

    def __init__(self, layout_id: int = 1, style_id: int = 1):
        from lightwheel_sdk.loader import floorplan_loader

        # Lazily download the USD
        self.usd_path = str(
            floorplan_loader.get_usd(
                scene="robocasakitchen", layout_id=layout_id, style_id=style_id, backend="robocasa"
            )[0]
        )
        super().__init__()


@register_asset
class RubiksCubeBowlBackgroundSRL(LibraryBackground):
    """
    A rubiks cube bowl background.
    """

    name = "rubiks_cube_bowl_srl"
    tags = ["background"]

    usd_path = "/datasets/assets/scenes/bowl_on_table.usda"
    object_min_z = -0.05
    spawn_cfg_addon = {"activate_contact_sensors": True}

    def __init__(self):
        super().__init__()


@register_asset
class TableMapleRobolab(LibraryBackground):
    """
    A maple table background.
    """

    name = "table_maple_robolab"
    tags = ["background"]
    usd_path = "/home/cvolk/datasets/assets/fixtures/table_maple.usd"
    object_min_z = -0.05

    def __init__(self):
        super().__init__()
