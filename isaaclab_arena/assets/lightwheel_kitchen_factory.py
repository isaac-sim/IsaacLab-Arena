# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Factory for Lightwheel RoboCasa kitchen background classes."""

from typing import Any

from isaaclab_arena.assets.lightwheel_kitchen_constants import LIGHTWHEEL_KITCHEN_LAYOUTS, LIGHTWHEEL_KITCHEN_STYLES
from isaaclab_arena.assets.register import register_asset


def register_lightwheel_kitchens(base_class: type[Any], namespace: dict[str, Any]) -> None:
    """Create and register every Lightwheel kitchen layout/style class.

    Args:
        base_class: Base class for generated kitchen backgrounds.
        namespace: Module namespace that exposes the generated classes.
    """
    for layout_id, layout_type, layout_name in LIGHTWHEEL_KITCHEN_LAYOUTS:
        for style_id, style_type, style_name in LIGHTWHEEL_KITCHEN_STYLES:
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
