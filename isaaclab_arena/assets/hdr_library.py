# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.hdr import HDR, TextureFormat
from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_hdr


class LibraryHDR(HDR):
    """Base class for HDRs in the library which are defined in this file."""

    name: str
    tags: list[str]
    texture_file: str
    texture_format: TextureFormat = "latlong"
    description: str = ""

    def __init__(self):
        super().__init__(
            name=self.name,
            texture_file=self.texture_file,
            tags=self.tags,
            description=self.description,
            texture_format=self.texture_format,
        )


@register_hdr
class HomeOfficeHDRRobolab(LibraryHDR):
    name = "home_office_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/home_office.exr"


@register_hdr
class EmptyWarehouseHDRRobolab(LibraryHDR):
    name = "empty_warehouse_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/empty_warehouse.hdr"


@register_hdr
class AerodynamicsWorkshopHDRRobolab(LibraryHDR):
    name = "aerodynamics_workshop_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/aerodynamics_workshop.hdr"


@register_hdr
class BilliardHallHDRRobolab(LibraryHDR):
    name = "billiard_hall_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/billiard_hall.hdr"


@register_hdr
class BrownPhotostudioHDRRobolab(LibraryHDR):
    name = "brown_photostudio_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/default/brown_photostudio.hdr"


@register_hdr
class BlindsHDRRobolab(LibraryHDR):
    name = "blinds_robolab"
    tags = ["indoor", "kitchen"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/blinds_2k.hdr"


@register_hdr
class KiaraInteriorHDRRobolab(LibraryHDR):
    name = "kiara_interior_robolab"
    tags = ["indoor", "kitchen"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/kiara_interior_2k.hdr"


@register_hdr
class GarageHDRRobolab(LibraryHDR):
    name = "garage_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/garage_2k.hdr"


@register_hdr
class WoodenLoungeHDRRobolab(LibraryHDR):
    name = "wooden_lounge_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/wooden_lounge_2k.hdr"


@register_hdr
class PhotoStudioHDRRobolab(LibraryHDR):
    name = "photo_studio_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/photo_studio_01_2k.hdr"


@register_hdr
class CarpentryShopHDRRobolab(LibraryHDR):
    name = "carpentry_shop_robolab"
    tags = ["indoor", "workshop"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/carpentry_shop_01_2k.hdr"
