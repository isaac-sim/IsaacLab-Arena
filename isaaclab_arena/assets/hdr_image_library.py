# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered HDR environment maps for dome lights."""

from isaaclab_arena.assets.hdr_image import HDRImage, TextureFormat
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_hdr


class LibraryHDR(HDRImage):
    """Base class for HDRs in the library which are defined in this file."""

    registry_name: str
    """Key of this HDR image class in the HDR image registry."""

    tags: list[str]
    texture_file: str
    texture_format: TextureFormat = "latlong"
    description: str = ""

    def __init__(self):
        super().__init__(
            name=self.registry_name,
            texture_file=self.texture_file,
            tags=self.tags,
            description=self.description,
            texture_format=self.texture_format,
        )


@register_hdr
class HomeOfficeHDRRobolab(LibraryHDR):
    """HDR environment map of a home office interior, from the Robolab assets."""

    registry_name = "home_office_robolab"
    tags = ["indoor"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/home_office.exr"
    )


@register_hdr
class EmptyWarehouseHDRRobolab(LibraryHDR):
    """HDR environment map of an empty warehouse, from the Robolab assets."""

    registry_name = "empty_warehouse_robolab"
    tags = ["indoor", "industrial"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/empty_warehouse.hdr"
    )


@register_hdr
class AerodynamicsWorkshopHDRRobolab(LibraryHDR):
    """HDR environment map of an aerodynamics workshop, from the Robolab assets."""

    registry_name = "aerodynamics_workshop_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/aerodynamics_workshop.hdr"


@register_hdr
class BilliardHallHDRRobolab(LibraryHDR):
    """HDR environment map of a billiard hall, from the Robolab assets."""

    registry_name = "billiard_hall_robolab"
    tags = ["indoor"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/billiard_hall.hdr"
    )


@register_hdr
class BrownPhotostudioHDRRobolab(LibraryHDR):
    """HDR environment map of a brown photo studio, from the Robolab assets."""

    registry_name = "brown_photostudio_robolab"
    tags = ["indoor", "studio"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/brown_photostudio.hdr"
    )


@register_hdr
class BlindsHDRRobolab(LibraryHDR):
    """HDR environment map of an interior with window blinds, from the Robolab assets."""

    registry_name = "blinds_robolab"
    tags = ["indoor", "kitchen"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/blinds_2k.hdr"
    )


@register_hdr
class KiaraInteriorHDRRobolab(LibraryHDR):
    """HDR environment map of an interior scene, from the Robolab assets."""

    registry_name = "kiara_interior_robolab"
    tags = ["indoor", "kitchen"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/kiara_interior_2k.hdr"
    )


@register_hdr
class GarageHDRRobolab(LibraryHDR):
    """HDR environment map of a garage, from the Robolab assets."""

    registry_name = "garage_robolab"
    tags = ["indoor"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/garage_2k.hdr"
    )


@register_hdr
class WoodenLoungeHDRRobolab(LibraryHDR):
    """HDR environment map of a wooden lounge, from the Robolab assets."""

    registry_name = "wooden_lounge_robolab"
    tags = ["indoor"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/wooden_lounge_2k.hdr"
    )


@register_hdr
class PhotoStudioHDRRobolab(LibraryHDR):
    """HDR environment map of a photo studio, from the Robolab assets."""

    registry_name = "photo_studio_robolab"
    tags = ["indoor", "studio"]
    texture_file = (
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/photo_studio_01_2k.hdr"
    )


@register_hdr
class CarpentryShopHDRRobolab(LibraryHDR):
    """HDR environment map of a carpentry shop, from the Robolab assets."""

    registry_name = "carpentry_shop_robolab"
    tags = ["indoor", "workshop"]
    texture_file = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/carpentry_shop_01_2k.hdr"
