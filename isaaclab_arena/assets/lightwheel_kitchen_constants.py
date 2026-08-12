# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightwheel RoboCasa kitchen layout and style metadata."""
# https://docs.lightwheel.net/lw_benchhub/task%20suites/Lightwheel%20Robocasa%20Tasks/Scenes/Kitchen%20Scenes#scene-configuration
LIGHTWHEEL_KITCHEN_LAYOUTS = (
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
"""Kitchen layouts as ``(layout_id, class_name_suffix, registry_name_suffix)`` tuples."""

LIGHTWHEEL_KITCHEN_STYLES = (
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
"""Kitchen styles as ``(style_id, class_name_suffix, registry_name_suffix)`` tuples."""
