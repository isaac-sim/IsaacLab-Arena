# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for Lightwheel RoboCasa kitchen background classes."""

from typing import Any

from isaaclab_arena.assets.register import register_asset

# https://docs.lightwheel.net/lw_benchhub/task%20suites/Lightwheel%20Robocasa%20Tasks/Scenes/Kitchen%20Scenes#scene-configuration
_LIGHTWHEEL_KITCHEN_LAYOUTS = (
    (1, "OneWall", "one_wall"),
    (2, "OneWallWithIsland", "one_wall_with_island"),
    (3, "LShaped", "l_shaped"),
    (4, "LShapedWithIsland", "l_shaped_with_island"),
    (5, "Galley", "galley"),
    (6, "UShaped", "u_shaped"),
    (7, "UShapedWithIsland", "u_shaped_with_island"),
    (8, "GShaped", "g_shaped"),
    (9, "GShapedLarge", "g_shaped_large"),
    (10, "Wraparound", "wraparound"),
)

_LIGHTWHEEL_KITCHEN_STYLES = (
    (1, "Coastal", "coastal"),
    (2, "Farmhouse1", "farmhouse1"),
    (3, "Industrial", "industrial"),
    (4, "Mediterranean", "mediterranean"),
    (5, "Modern1", "modern1"),
    (6, "Modern2", "modern2"),
    (7, "Rustic", "rustic"),
    (8, "Scandinavian", "scandinavian"),
    (9, "Traditional", "traditional"),
    (10, "Farmhouse2", "farmhouse2"),
)


def register_lightwheel_kitchens(base_class: type[Any], namespace: dict[str, Any]) -> None:
    """Create and register every Lightwheel kitchen layout/style class.

    Args:
        base_class: Base class for generated kitchen backgrounds.
        namespace: Module namespace that exposes the generated classes.
    """
    for layout_id, layout_type, layout_name in _LIGHTWHEEL_KITCHEN_LAYOUTS:
        for style_id, style_type, style_name in _LIGHTWHEEL_KITCHEN_STYLES:
            class_name = f"LightwheelKitchen{layout_type}{style_type}"
            background_class = type(
                class_name,
                (base_class,),
                {
                    "__module__": base_class.__module__,
                    "__doc__": f"Lightwheel RoboCasa {layout_name} kitchen, style {style_id}.",
                    "name": f"lightwheel_kitchen_{layout_name}_{style_name}",
                    "layout_id": layout_id,
                    "style_id": style_id,
                },
            )
            namespace[class_name] = register_asset(background_class)
