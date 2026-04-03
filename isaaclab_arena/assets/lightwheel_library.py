# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-registered Lightwheel assets from local cache.

Scans ~/.cache/lightwheel_sdk/object/ and registers each CamelCase asset
as a LibraryObject. ObjectType is auto-detected from the USD.
When both CamelCase and snake_case variants exist for a category,
only CamelCase variants are kept.
"""

from __future__ import annotations

import glob
import os
import re

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

CACHE_DIR = os.path.expanduser("~/.cache/lightwheel_sdk/object")

# Large furniture, fixtures, and layout elements to skip
SKIP_CATEGORIES = {
    "cabinet",
    "cabinet_door_panel",
    "cabinet_handle",
    "coffee_machine",
    "decoration",
    "dish_rack",
    "dishwasher",
    "electric_kettle",
    "faucet",
    "floor_lamp",
    "floor_layout",
    "oven",
    "oven_tray",
    "storage_box",
    "tray",
    "range_hood",
    "refrigerator",
    "shelf",
    "sink",
    "socket",
    "sofa",
    "stand_mixer",
    "stool",
    "storage_furniture",
    "stove",
    "stovetop",
    "switch",
    "table",
    "television",
    "toaster_oven",
    "utensil_rack",
    "utensil_set",
    "wall_layout",
    "window",
    "wine_rack",
}


def _get_base_category(name: str) -> str:
    if name[0].isupper():
        base = re.sub(r'\d+$', '', name)
        base = re.sub(r'(?<!^)(?=[A-Z])', '_', base).lower()
    else:
        base = re.sub(r'_?\d+$', '', name)
    return base


def _scan_and_dedup() -> list[dict]:
    """Scan cache, return CamelCase-preferred asset list."""
    if not os.path.isdir(CACHE_DIR):
        return []

    all_assets = []
    for entry in sorted(os.listdir(CACHE_DIR)):
        entry_path = os.path.join(CACHE_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        usd_path = os.path.join(entry_path, f"{entry}.usd")
        if not os.path.exists(usd_path):
            usd_files = glob.glob(os.path.join(entry_path, "*.usd"))
            if usd_files:
                usd_path = usd_files[0]
            else:
                continue
        all_assets.append({"file_name": entry, "usd_path": usd_path})

    # Group by category and prefer CamelCase
    by_category = {}
    for a in all_assets:
        cat = _get_base_category(a["file_name"])
        by_category.setdefault(cat, []).append(a)

    deduped = []
    for cat, variants in by_category.items():
        if cat in SKIP_CATEGORIES:
            continue
        camel = [v for v in variants if v["file_name"][0].isupper()]
        snake = [v for v in variants if v["file_name"][0].islower()]
        deduped.extend(camel if camel else snake)

    return sorted(deduped, key=lambda a: a["file_name"])


def _make_library_class(file_name: str, usd_path: str) -> type:
    """Dynamically create and register a LibraryObject subclass."""
    # object_type=None triggers auto-detection from the USD
    cls = type(
        file_name,
        (LibraryObject,),
        {
            "name": file_name,
            "tags": ["object", "lightwheel"],
            "usd_path": usd_path,
            "object_type": None,  # auto-detect RIGID vs ARTICULATION
        },
    )
    return cls


# Auto-register all assets on import
_lightwheel_assets = _scan_and_dedup()
_registered_classes = {}

for _asset in _lightwheel_assets:
    _cls = _make_library_class(_asset["file_name"], _asset["usd_path"])
    _cls = register_asset(_cls)
    _registered_classes[_asset["file_name"]] = _cls


def get_all_lightwheel_assets() -> dict[str, type]:
    """Return dict of {file_name: registered class} for all Lightwheel assets."""
    return dict(_registered_classes)


def get_lightwheel_asset(file_name: str) -> type:
    """Get a specific Lightwheel asset class by file_name."""
    return _registered_classes[file_name]
