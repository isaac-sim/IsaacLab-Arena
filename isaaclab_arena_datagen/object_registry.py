# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Object type classification and instance identity tracking."""

from __future__ import annotations

import colorsys
import enum
from dataclasses import dataclass

from isaaclab_arena_datagen.utils.constants import GOLDEN_RATIO_CONJUGATE, HUE_OFFSET, get_type_prefix_map


class ObjectType(enum.IntEnum):
    """Classification of a tracked object in the scene."""

    STATIC = 0
    RIGID = 1
    ARTICULATION = 2
    UNSUPPORTED = 255

    @property
    def label(self) -> str:
        """Return the lowercase name of this member (e.g. ``"rigid"``)."""
        return self.name.lower()


@dataclass(frozen=True)
class InstanceKey:
    """Hashable identity for a tracked object instance.

    Used as a dict/set key to provide temporally consistent IDs, names,
    and colours across cameras and time steps.
    """

    kind: ObjectType
    asset_name: str


class ObjectInstanceRegistry:
    """Shared registry that assigns temporally consistent object IDs, names, and colors.

    Pass a single instance to every :class:`IsaacLabArenaCameraHandler` so that
    the same physical object receives the same ``object_id``, ``object_name``,
    and RGBA color regardless of which camera observes it first.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._color_to_object_id: dict[tuple, int] = {}
        self._next_object_id: int = 0

        self._instance_key_to_object_id: dict[InstanceKey, int] = {}
        self._instance_key_to_name: dict[InstanceKey, str] = {}
        self._next_instance_object_id: int = 0
        self._next_instance_idx: dict[ObjectType, int] = {t: 1 for t in ObjectType}

    # -- colour-based legacy ID (used by get_semantic_info) ----------------

    def get_object_id(self, rgba: tuple) -> int:
        """Return a stable integer object ID for a given RGBA colour."""
        if rgba not in self._color_to_object_id:
            self._color_to_object_id[rgba] = self._next_object_id
            self._next_object_id += 1
        return self._color_to_object_id[rgba]

    # -- instance-key-based identity (used by get_object_instance_segmentation) --

    @staticmethod
    def _safe_name_token(raw: str) -> str:
        """Convert an arbitrary label/path to an ASCII-safe token.

        Replaces non-alphanumeric characters with underscores and strips
        leading/trailing underscores.  Returns ``"unknown"`` for empty input.
        """
        token = "".join(ch if ch.isalnum() else "_" for ch in raw.strip("/"))
        token = token.strip("_")
        return token or "unknown"

    def instance_key_to_display_name(self, instance_key: InstanceKey) -> str:
        """Get or create a temporally consistent display name for an object key.

        Display names follow the pattern ``{type_prefix}_{index}_{asset_token}``
        (e.g. ``rigid_object_1_cube``).  The special ``background`` static
        object always receives the name ``"background"``.
        """
        if instance_key in self._instance_key_to_name:
            return self._instance_key_to_name[instance_key]

        kind = instance_key.kind
        asset_name = instance_key.asset_name

        if kind is ObjectType.STATIC and asset_name == "background":
            name = "background"
        else:
            type_prefix = get_type_prefix_map()[kind]
            idx = self._next_instance_idx[kind]
            self._next_instance_idx[kind] = idx + 1
            raw = asset_name.split("/")[-1] if kind in (ObjectType.STATIC, ObjectType.UNSUPPORTED) else asset_name
            suffix = self._safe_name_token(raw)
            name = f"{type_prefix}_{idx}_{suffix}"

        self._instance_key_to_name[instance_key] = name
        return name

    @staticmethod
    def _instance_id_to_rgba(object_id: int) -> tuple[int, int, int, int]:
        """Deterministically map an object ID to a vivid RGBA color.

        Uses the golden-ratio-conjugate method to produce maximally-spaced
        hues across the colour wheel.
        """
        hue = (HUE_OFFSET + GOLDEN_RATIO_CONJUGATE * object_id) % 1.0
        sat = 0.75
        val = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 255)

    def get_or_create_instance_identity(self, instance_key: InstanceKey) -> tuple[int, str, tuple[int, int, int, int]]:
        """Allocate/retrieve stable object-id, name, and color for an instance key."""
        if instance_key not in self._instance_key_to_object_id:
            self._instance_key_to_object_id[instance_key] = self._next_instance_object_id
            self._next_instance_object_id += 1
        object_id = self._instance_key_to_object_id[instance_key]
        object_name = self.instance_key_to_display_name(instance_key)
        rgba = self._instance_id_to_rgba(object_id)
        return object_id, object_name, rgba
