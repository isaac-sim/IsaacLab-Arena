# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.hdr_image import HDRImage, TextureFormat
from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_hdr


class LibraryHDR(HDRImage):
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


# ============================================================
# RoboLab HDR backgrounds (auto-generated) — 89 HDRs
# ============================================================


@register_hdr
class AftLounge2kHDRRobolab(LibraryHDR):
    name = "aft_lounge_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/aft_lounge_2k.hdr"
    description = "lounge,chair,ship,stage,carpet,window,ocean,sun,lamp,table,indoor,urban,sunrise-sunset,clear,high contrast,natural light,artificial light"


@register_hdr
class AircraftWorkshop012kHDRRobolab(LibraryHDR):
    name = "aircraft_workshop_01_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/aircraft_workshop_01_2k.hdr"
    description = "workshop,industrial,warehouse,scaffold,indoor,urban,morning-afternoon,medium contrast,natural light,artificial light"


@register_hdr
class AnniversaryLounge2kHDRRobolab(LibraryHDR):
    name = "anniversary_lounge_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/anniversary_lounge_2k.hdr"
    description = "window,room,couch,lamp,carpet,table,indoor,urban,medium contrast,natural light,artificial light"


@register_hdr
class ArtStudio2kHDRRobolab(LibraryHDR):
    name = "art_studio_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/art_studio_2k.hdr"
    description = "painting,art,workshop,window,indoor,urban,medium contrast,natural light,artificial light"


@register_hdr
class ArtistWorkshop2kHDRRobolab(LibraryHDR):
    name = "artist_workshop_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/artist_workshop_2k.hdr"
    description = "painting,canvas,window,art,indoor,urban,midday,low contrast,natural light"


@register_hdr
class AutoService2kHDRRobolab(LibraryHDR):
    name = "auto_service_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/auto_service_2k.hdr"
    description = "workshop,warehouse,industrial,concrete,indoor,urban,medium contrast,artificial light"


@register_hdr
class Autoshop012kHDRRobolab(LibraryHDR):
    name = "autoshop_01_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/autoshop_01_2k.hdr"
    description = "garage,warehouse,industrial,car,workshop,indoor,urban,low contrast,artificial light"


@register_hdr
class Ballroom2kHDRRobolab(LibraryHDR):
    name = "ballroom_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/ballroom_2k.hdr"
    description = "tile,victorian,window,piano,chandelier,indoor,urban,medium contrast,artificial light,natural light"


@register_hdr
class Bathroom2kHDRRobolab(LibraryHDR):
    name = "bathroom_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/bathroom_2k.hdr"
    description = "bath,basin,tiles,sink,indoor,urban,medium contrast,artificial light"


@register_hdr
class BlenderInstitute2kHDRRobolab(LibraryHDR):
    name = "blender_institute_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/blender_institute_2k.hdr"
    description = "blender,timber,computer,chair,lamp,people,indoor,urban,low contrast,artificial light"


@register_hdr
class BoilerRoom2kHDRRobolab(LibraryHDR):
    name = "boiler_room_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/boiler_room_2k.hdr"
    description = "machine,industrial,pipes,factory,indoor,urban,low contrast,natural light,artificial light"


@register_hdr
class BushRestaurant2kHDRRobolab(LibraryHDR):
    name = "bush_restaurant_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/bush_restaurant_2k.hdr"
    description = "table,chair,restaurant,window,door,indoor,urban,medium contrast,natural light"


@register_hdr
class Cabin2kHDRRobolab(LibraryHDR):
    name = "cabin_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/cabin_2k.hdr"
    description = "bed,tv,hotel,cabin,boat,ship,cruise,indoor,urban,high contrast,artificial light"


@register_hdr
class CapeHill2kHDRRobolab(LibraryHDR):
    name = "cape_hill_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/cape_hill_2k.hdr"
    description = "rock,shrub,view,ocean,hill,hilltop,sun,outdoor,skies,nature,sunrise-sunset,partly cloudy,high contrast,natural light"


@register_hdr
class CarpentryShop022kHDRRobolab(LibraryHDR):
    name = "carpentry_shop_02_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/carpentry_shop_02_2k.hdr"
    description = "lamp,fluorescent,workshop,indoor,urban,high contrast,artificial light"


@register_hdr
class CayleyInterior2kHDRRobolab(LibraryHDR):
    name = "cayley_interior_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/cayley_interior_2k.hdr"
    description = "lamp,table,window,glass,balcony,view,log,indoor,urban,sunrise-sunset,artificial light,natural light,high contrast"


@register_hdr
class ChildrensHospital2kHDRRobolab(LibraryHDR):
    name = "childrens_hospital_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/childrens_hospital_2k.hdr"
    description = "bed,ward,medical,hospital,room,indoor,urban,high contrast,artificial light"


@register_hdr
class CinemaHall2kHDRRobolab(LibraryHDR):
    name = "cinema_hall_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/cinema_hall_2k.hdr"
    description = "movie,cinema,chair,seat,projector,theater,indoor,urban,medium contrast,artificial light"


@register_hdr
class CinemaLobby2kHDRRobolab(LibraryHDR):
    name = "cinema_lobby_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/cinema_lobby_2k.hdr"
    description = "couch,seat,bench,fluorescent,indoor,urban,medium contrast,artificial light"


@register_hdr
class CircusArena2kHDRRobolab(LibraryHDR):
    name = "circus_arena_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/circus_arena_2k.hdr"
    description = "seat,theater,arena,show,indoor,urban,medium contrast,artificial light"


@register_hdr
class ColorfulStudio2kHDRRobolab(LibraryHDR):
    name = "colorful_studio_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/colorful_studio_2k.hdr"
    description = "couch,bar,bottle,incandescent,set,lounge,indoor,urban,high contrast,artificial light"


@register_hdr
class CombinationRoom2kHDRRobolab(LibraryHDR):
    name = "combination_room_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/combination_room_2k.hdr"
    description = "couch,victorian,curtain,lamp,painting,window,indoor,urban,medium contrast,natural light,artificial light"


@register_hdr
class ConcreteTunnel022kHDRRobolab(LibraryHDR):
    name = "concrete_tunnel_02_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/concrete_tunnel_02_2k.hdr"
    description = "asphalt,concrete,tunnel,road,underpass,indoor,urban,midday,medium contrast,natural light"


@register_hdr
class ConcreteTunnel2kHDRRobolab(LibraryHDR):
    name = "concrete_tunnel_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/concrete_tunnel_2k.hdr"
    description = "tunnel,brick,concrete,lamp,bench,indoor,urban,high contrast,artificial light"


@register_hdr
class DeBalie2kHDRRobolab(LibraryHDR):
    name = "de_balie_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/de_balie_2k.hdr"
    description = "stair,banister,people,bcon,door,meeting,pink,indoor,urban,medium contrast,artificial light"


@register_hdr
class DresdenStationNight2kHDRRobolab(LibraryHDR):
    name = "dresden_station_night_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/dresden_station_night_2k.hdr"
    description = "train,station,bench,hall,tracks,fluorescent,indoor,urban,night,medium contrast,artificial light"


@register_hdr
class EnSuite2kHDRRobolab(LibraryHDR):
    name = "en_suite_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/en_suite_2k.hdr"
    description = "bathroom,sink,basin,shower,toilet,indoor,urban,artificial light,high contrast"


@register_hdr
class EntranceHall2kHDRRobolab(LibraryHDR):
    name = "entrance_hall_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/entrance_hall_2k.hdr"
    description = "victorian,window,tile,indoor,urban,midday,medium contrast,natural light,artificial light"


@register_hdr
class Fireplace2kHDRRobolab(LibraryHDR):
    name = "fireplace_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/fireplace_2k.hdr"
    description = "fire,couch,room,lounge,indoor,urban,night,high contrast,artificial light"


@register_hdr
class FloralTent2kHDRRobolab(LibraryHDR):
    name = "floral_tent_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/floral_tent_2k.hdr"
    description = "bush,plant,sun,tiles,tent,canopy,indoor,nature,urban,midday,medium contrast,natural light"


@register_hdr
class Georgentor2kHDRRobolab(LibraryHDR):
    name = "georgentor_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/georgentor_2k.hdr"
    description = "brick,architecture,tunnel,passage,indoor,urban,medium contrast,natural light"


@register_hdr
class GlassPassage2kHDRRobolab(LibraryHDR):
    name = "glass_passage_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/glass_passage_2k.hdr"
    description = "tunnel,plant,tree,lamp,enclosure,indoor,nature,urban,low contrast,natural light,artificial light"


@register_hdr
class Gym012kHDRRobolab(LibraryHDR):
    name = "gym_01_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/gym_01_2k.hdr"
    description = "fluorescent,gym,indoor,urban,medium contrast,artificial light"


@register_hdr
class GymEntrance2kHDRRobolab(LibraryHDR):
    name = "gym_entrance_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/gym_entrance_2k.hdr"
    description = "shop,couch,gym,red,fluorescent,indoor,urban,medium contrast,artificial light"


@register_hdr
class HallOfFinfish2kHDRRobolab(LibraryHDR):
    name = "hall_of_finfish_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/hall_of_finfish_2k.hdr"
    description = "fish,shark,museum,indoor,urban,low contrast,artificial light"


@register_hdr
class HallOfMammals2kHDRRobolab(LibraryHDR):
    name = "hall_of_mammals_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/hall_of_mammals_2k.hdr"
    description = "museum,bone,lion,tiger,whale,skeleton,indoor,urban,medium contrast,artificial light"


@register_hdr
class HospitalRoom2kHDRRobolab(LibraryHDR):
    name = "hospital_room_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/hospital_room_2k.hdr"
    description = "bed,table,hospital,lamp,medical,operation,indoor,urban,medium contrast,artificial light"


@register_hdr
class HotelRoom2kHDRRobolab(LibraryHDR):
    name = "hotel_room_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/hotel_room_2k.hdr"
    description = "bed,tv,table,chair,mirror,window,curtain,lamp,hotel,indoor,urban,high contrast,natural light,artificial light"


@register_hdr
class ImmenstadterHorn2kHDRRobolab(LibraryHDR):
    name = "immenstadter_horn_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/immenstadter_horn_2k.hdr"
    description = "sun,tree,grass,hilltop,hill,fence,outdoor,nature,midday,partly cloudy,medium contrast,natural light"


@register_hdr
class IndoorPool2kHDRRobolab(LibraryHDR):
    name = "indoor_pool_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/indoor_pool_2k.hdr"
    description = "pool,water,lamp,chair,indoor,urban,low contrast,artificial light"


@register_hdr
class IndustrialPipeAndValve012kHDRRobolab(LibraryHDR):
    name = "industrial_pipe_and_valve_01_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/industrial_pipe_and_valve_01_2k.hdr"
    description = "workshop,lamp,industrial,warehouse,indoor,urban,high contrast,artificial light"


@register_hdr
class IndustrialPipeAndValve022kHDRRobolab(LibraryHDR):
    name = "industrial_pipe_and_valve_02_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/industrial_pipe_and_valve_02_2k.hdr"
    description = "industrial,lamp,warehouse,workshop,artificial light,high contrast,indoor,urban"


@register_hdr
class Lapa2kHDRRobolab(LibraryHDR):
    name = "lapa_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/lapa_2k.hdr"
    description = "thatch,couch,garden,plant,gazebo,indoor,urban,medium contrast,natural light,artificial light"


@register_hdr
class LeadenhallMarket2kHDRRobolab(LibraryHDR):
    name = "leadenhall_market_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/leadenhall_market_2k.hdr"
    description = "hall,mall,europe,architecture,london,cobblestone,street,road,indoor,urban,morning-afternoon,medium contrast,natural light,artificial light"


@register_hdr
class Lebombo2kHDRRobolab(LibraryHDR):
    name = "lebombo_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/lebombo_2k.hdr"
    description = "house,lamp,window,wood floor,empty,indoor,urban,morning-afternoon,low contrast,natural light,artificial light"


@register_hdr
class Lookout2kHDRRobolab(LibraryHDR):
    name = "lookout_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/lookout_2k.hdr"
    description = "thatch,hide,pilanesberg,window,indoor,urban,nature,medium contrast,natural light"


@register_hdr
class LythwoodLounge2kHDRRobolab(LibraryHDR):
    name = "lythwood_lounge_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/lythwood_lounge_2k.hdr"
    description = "couch,warm,lounge,lamp,room,indoor,urban,midday,overcast,medium contrast,natural light,artificial light"


@register_hdr
class LythwoodRoom2kHDRRobolab(LibraryHDR):
    name = "lythwood_room_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/lythwood_room_2k.hdr"
    description = "couch,bed,lamp,window,mantel,carpet,indoor,urban,midday,overcast,low contrast,natural light,artificial light"


@register_hdr
class MachineShop012kHDRRobolab(LibraryHDR):
    name = "machine_shop_01_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/machine_shop_01_2k.hdr"
    description = "workshop,hall,warehouse,industrial,machine,metal,indoor,urban,medium contrast,natural light"


@register_hdr
class MachineShop022kHDRRobolab(LibraryHDR):
    name = "machine_shop_02_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/machine_shop_02_2k.hdr"
    description = "workshop,hall,warehouse,industrial,machine,metal,indoor,urban,medium contrast,natural light"


@register_hdr
class MachineShop032kHDRRobolab(LibraryHDR):
    name = "machine_shop_03_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/machine_shop_03_2k.hdr"
    description = "hall,industrial,machine,warehouse,workshop,indoor,urban,morning-afternoon,medium contrast,natural light"


@register_hdr
class MuseumOfEthnography2kHDRRobolab(LibraryHDR):
    name = "museum_of_ethnography_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/museum_of_ethnography_2k.hdr"
    description = "museum,antique,display,indoor,urban,medium contrast,artificial light"


@register_hdr
class MusicHall012kHDRRobolab(LibraryHDR):
    name = "music_hall_01_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/music_hall_01_2k.hdr"
    description = "theater,organ,chair,victorian,hall,indoor,urban,medium contrast,artificial light"


@register_hdr
class MusicHall022kHDRRobolab(LibraryHDR):
    name = "music_hall_02_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/music_hall_02_2k.hdr"
    description = "hall,empty,victorian,window,couch,indoor,urban,medium contrast,artificial light,natural light"


@register_hdr
class Mutianyu2kHDRRobolab(LibraryHDR):
    name = "mutianyu_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/mutianyu_2k.hdr"
    description = "arch,brick,window,mountain,great wall,indoor,urban,morning-afternoon,clear,low contrast,natural light"


@register_hdr
class OldApartmentsWalkway2kHDRRobolab(LibraryHDR):
    name = "old_apartments_walkway_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/old_apartments_walkway_2k.hdr"
    description = "tree,railing,abandoned,balcony,indoor,urban,midday,clear,low contrast,natural light"


@register_hdr
class OldBusDepot2kHDRRobolab(LibraryHDR):
    name = "old_bus_depot_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/old_bus_depot_2k.hdr"
    description = "warehouse,industrial,concrete,abandoned,indoor,urban,midday,medium contrast,natural light"


@register_hdr
class OldDepot2kHDRRobolab(LibraryHDR):
    name = "old_depot_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/old_depot_2k.hdr"
    description = "warehouse,abandoned,industrial,indoor,urban,morning-afternoon,clear,low contrast,natural light"


@register_hdr
class OldOutdoorTheater2kHDRRobolab(LibraryHDR):
    name = "old_outdoor_theater_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/old_outdoor_theater_2k.hdr"
    description = "sun,arena,europe,outdoor,skies,urban,midday,partly cloudy,high contrast,natural light"


@register_hdr
class ParkingGarage2kHDRRobolab(LibraryHDR):
    name = "parking_garage_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/parking_garage_2k.hdr"
    description = "lamp,car,parking,road,garage,underground,indoor,urban,natural light,high contrast"


@register_hdr
class PeppermintPowerplant2kHDRRobolab(LibraryHDR):
    name = "peppermint_powerplant_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/peppermint_powerplant_2k.hdr"
    description = "abandoned,industrial,factory,workshop,indoor,urban,morning-afternoon,medium contrast,natural light"


@register_hdr
class PhoneShop2kHDRRobolab(LibraryHDR):
    name = "phone_shop_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/phone_shop_2k.hdr"
    description = "lamp,shop,store,display,counter,indoor,urban,medium contrast,artificial light"


@register_hdr
class Pillars2kHDRRobolab(LibraryHDR):
    name = "pillars_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/pillars_2k.hdr"
    description = "column,architecture,monument,indoor,urban,morning-afternoon,low contrast,natural light"


@register_hdr
class PumpHouse2kHDRRobolab(LibraryHDR):
    name = "pump_house_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/pump_house_2k.hdr"
    description = "brick,pump,dirt,pipe,shed,window,indoor,urban,morning-afternoon,high contrast,natural light"


@register_hdr
class ReadingRoom2kHDRRobolab(LibraryHDR):
    name = "reading_room_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/reading_room_2k.hdr"
    description = "chair,couch,window,door,tile,indoor,urban,morning-afternoon,medium contrast,natural light"


@register_hdr
class RoyalEsplanade2kHDRRobolab(LibraryHDR):
    name = "royal_esplanade_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/royal_esplanade_2k.hdr"
    description = "mall,cruise,screen,balcony,stair,tile,indoor,urban,high contrast,artificial light"


@register_hdr
class SculptureExhibition2kHDRRobolab(LibraryHDR):
    name = "sculpture_exhibition_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/sculpture_exhibition_2k.hdr"
    description = "horse,art,exhibition,museum,indoor,urban,medium contrast,artificial light"


@register_hdr
class SmallCathedral2kHDRRobolab(LibraryHDR):
    name = "small_cathedral_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/small_cathedral_2k.hdr"
    description = "church,chapel,carpet,painting,religion,indoor,urban,low contrast,natural light"


@register_hdr
class SmallEmptyHouse2kHDRRobolab(LibraryHDR):
    name = "small_empty_house_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/small_empty_house_2k.hdr"
    description = "house,tile,lamp,door,indoor,urban,morning-afternoon,medium contrast,natural light"


@register_hdr
class SmallHangar012kHDRRobolab(LibraryHDR):
    name = "small_hangar_01_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/small_hangar_01_2k.hdr"
    description = "hangar,plane,industrial,warehouse,indoor,urban,morning-afternoon,low contrast,natural light"


@register_hdr
class SmallHangar022kHDRRobolab(LibraryHDR):
    name = "small_hangar_02_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/small_hangar_02_2k.hdr"
    description = "hangar,plane,industrial,europe,indoor,urban,morning-afternoon,medium contrast,natural light"


@register_hdr
class StFagansInterior2kHDRRobolab(LibraryHDR):
    name = "st_fagans_interior_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/st_fagans_interior_2k.hdr"
    description = "bench,couch,lobby,modern,museum,stairs,wooden floor,indoor,urban,low contrast,artificial light"


@register_hdr
class Storeroom2kHDRRobolab(LibraryHDR):
    name = "storeroom_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/storeroom_2k.hdr"
    description = "mess,junk,fluorescent,indoor,urban,high contrast,artificial light"


@register_hdr
class StudioSmall012kHDRRobolab(LibraryHDR):
    name = "studio_small_01_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_01_2k.hdr"
    description = "lamp,studio,photo,indoor,urban,studio,high contrast,artificial_light"


@register_hdr
class StudioSmall022kHDRRobolab(LibraryHDR):
    name = "studio_small_02_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_02_2k.hdr"
    description = "lamp,studio,photo,indoor,urban,studio,high contrast,artificial_light"


@register_hdr
class StudioSmall032kHDRRobolab(LibraryHDR):
    name = "studio_small_03_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_03_2k.hdr"
    description = "lamp,studio,photo,umbrella,indoor,urban,studio,high contrast,artificial_light"


@register_hdr
class StudioSmall042kHDRRobolab(LibraryHDR):
    name = "studio_small_04_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_04_2k.hdr"
    description = "lamp,studio,photo,indoor,studio,urban,high contrast,artificial_light"


@register_hdr
class StudioSmall052kHDRRobolab(LibraryHDR):
    name = "studio_small_05_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_05_2k.hdr"
    description = "lamp,studio,photo,reflector,dish,indoor,urban,studio,high contrast,artificial_light"


@register_hdr
class StudioSmall062kHDRRobolab(LibraryHDR):
    name = "studio_small_06_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_06_2k.hdr"
    description = "lamp,studio,photo,indoor,urban,studio,high contrast,artificial_light"


@register_hdr
class StudioSmall072kHDRRobolab(LibraryHDR):
    name = "studio_small_07_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/studio_small_07_2k.hdr"
    description = "lamp,photo,studio,single,indoor,studio,urban,high contrast,artificial light"


@register_hdr
class SubwayEntrance2kHDRRobolab(LibraryHDR):
    name = "subway_entrance_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/subway_entrance_2k.hdr"
    description = "tile,fluorescent,stair,map,tunnel,station,escalator,indoor,urban,midday,medium contrast,artificial light"


@register_hdr
class Surgery2kHDRRobolab(LibraryHDR):
    name = "surgery_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/surgery_2k.hdr"
    description = "bed,table,hospital,lamp,medical,operation,indoor,urban,medium contrast,artificial light"


@register_hdr
class Theater012kHDRRobolab(LibraryHDR):
    name = "theater_01_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/theater_01_2k.hdr"
    description = "theater,theatre,seat,orchestra,stage,curtain,indoor,urban,medium contrast,artificial light"


@register_hdr
class Theater022kHDRRobolab(LibraryHDR):
    name = "theater_02_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/theater_02_2k.hdr"
    description = "theater,theatre,seat,orchestra,stage,curtain,indoor,urban,medium contrast,artificial light"


@register_hdr
class TvStudio2kHDRRobolab(LibraryHDR):
    name = "tv_studio_2k_robolab"
    tags = ["indoor", "studio"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/tv_studio_2k.hdr"
    description = "camera,studio,set,news,show,indoor,urban,high contrast,artificial light"


@register_hdr
class VintageMeasuringLab2kHDRRobolab(LibraryHDR):
    name = "vintage_measuring_lab_2k_robolab"
    tags = ["indoor", "industrial"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/vintage_measuring_lab_2k.hdr"
    description = "machine,industrial,workshop,laboratory,indoor,urban,medium contrast,natural light,artificial light"


@register_hdr
class VultureHide2kHDRRobolab(LibraryHDR):
    name = "vulture_hide_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/vulture_hide_2k.hdr"
    description = "hide,bench,brick,window,indoor,urban,medium contrast,natural light"


@register_hdr
class WhaleSkeleton2kHDRRobolab(LibraryHDR):
    name = "whale_skeleton_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/whale_skeleton_2k.hdr"
    description = "museum,bone,whale,skeleton,indoor,urban,medium contrast,artificial light"


@register_hdr
class YarisInteriorGarage2kHDRRobolab(LibraryHDR):
    name = "yaris_interior_garage_2k_robolab"
    tags = ["indoor"]
    texture_file = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/backgrounds/indoors/yaris_interior_garage_2k.hdr"
    description = "car,driver,seat,garage,indoor,urban,midday,medium contrast,natural light,artificial light"