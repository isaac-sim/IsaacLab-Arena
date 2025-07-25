# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.scene.scene_registry import ObjectRegistry


def get_scene_details(background_name: str, pick_up_object_name: str):
    object_registry = ObjectRegistry()
    if background_name:
        background = object_registry.get_object_by_name(background_name)
    else:
        background = object_registry.get_random_object_by_tag("background")
    if pick_up_object_name:
        pick_up_object = object_registry.get_object_by_name(pick_up_object_name)
    else:
        pick_up_object = object_registry.get_random_object_by_tag("pick_up_object")
    # Add the initial state of the pick up object
    if pick_up_object:
        pick_up_object.get_pick_up_object().init_state = background.get_pick_up_object_location()
    scene_details = {
        "background": background.get_background(),
        "pick_up_object": pick_up_object.get_pick_up_object(),
        "destination_object": background.get_destination(),
    }
    return scene_details
