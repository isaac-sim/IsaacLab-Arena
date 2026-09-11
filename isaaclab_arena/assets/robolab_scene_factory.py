# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register full-scene RoboLab backgrounds used by the exact task catalog."""

from pathlib import Path
from typing import Any

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR

ROBOLAB_EXACT_SCENE_NAMES = (
    "bagel_plate_banana_bowl",
    "banana_bowl",
    "bin_condiments",
    "bin_mug_mustard_marker_bowl",
    "bottles_crate",
    "butter_raisin_box",
    "butter_raisin_box_grey_bin",
    "clutter_fruit_bottle_bluebin",
    "foodpacking_1bin_1box_1can",
    "mustard_raisin_box",
    "rubiks_cube_banana_bowl",
    "rubiks_cube_bowl",
    "tools_container",
    "two_bin",
    "workdesk",
    "workdesk_bin",
    "workdesk_snacks",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_SCENE_DIR = _REPO_ROOT / "RoboLab" / "assets" / "scenes"


def _scene_usd_path(scene_name: str) -> str:
    """Use the source checkout when present, otherwise the packaged Nucleus mirror."""
    local_path = _LOCAL_SCENE_DIR / f"{scene_name}.usda"
    if local_path.exists():
        return str(local_path)
    return f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/scenes/{scene_name}.usda"


def register_robolab_exact_scenes(base_class: type, namespace: dict[str, Any]) -> None:
    """Register scene-specific backgrounds backed by mirrored RoboLab USDAs."""
    from isaaclab_arena.assets.register import register_asset

    for scene_name in ROBOLAB_EXACT_SCENE_NAMES:
        registry_name = f"{scene_name}_robolab_exact"
        class_name = "".join(part.title() for part in registry_name.split("_"))
        scene_class = type(
            class_name,
            (base_class,),
            {
                "__module__": base_class.__module__,
                "name": registry_name,
                "tags": ["background", "robolab", "exact"],
                "usd_path": _scene_usd_path(scene_name),
                "object_min_z": -0.05,
                "reset_nested_physics": True,
            },
        )
        namespace[class_name] = register_asset(scene_class)
