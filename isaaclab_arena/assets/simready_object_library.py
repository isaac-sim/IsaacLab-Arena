# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic SimReady USD object registered for agent-generated environments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.simready_constants import (
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_SEARCH_REGISTRY_PREFIX,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
)
from isaaclab_arena.utils.pose import Pose

_NON_IDENTIFIER_CHARACTERS = re.compile(r"[^a-z0-9]+")


@register_asset
class SimReadyUsdObject(Object):
    """Spawn a SimReady asset from an explicit USD path supplied in graph params."""

    name = SIMREADY_USD_OBJECT_REGISTRY_NAME

    tags = ["sim-ready"]

    object_type = ObjectType.RIGID
    """The search only accepts an asset with exactly one rigid body."""

    def __init__(
        self,
        usd_path: str,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ):
        assert usd_path, "simready_usd_object requires params.usd_path"
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        spawn_cfg_addon.setdefault("variants", dict(SIMREADY_PHYSICS_VARIANTS))
        super().__init__(
            name=instance_name or self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


def simready_search_registry_name(search_phrase: str) -> str:
    """Return the catalogue name used to register the SimReady asset found for a search phrase."""
    slug = _NON_IDENTIFIER_CHARACTERS.sub("_", search_phrase.strip().lower()).strip("_")
    assert slug, f"search phrase {search_phrase!r} has no characters usable in a registry name"
    return f"{SIMREADY_SEARCH_REGISTRY_PREFIX}{slug}"


def register_searched_simready_object(
    search_phrase: str,
    usd_path: str,
    tags: Sequence[str] = (),
) -> type[SimReadyUsdObject]:
    """Register a searched SimReady asset so it can be picked like any other catalogue object.

    Args:
        search_phrase: Phrase the asset was found for; it determines the catalogue name.
        usd_path: USD path or URL of the found asset.
        tags: Extra catalogue tags to expose alongside the SimReady ones.

    Returns:
        The registered asset class.
    """
    registry_name = simready_search_registry_name(search_phrase)
    registry = AssetRegistry()
    if registry.is_registered(registry_name, ensure_loaded=False):
        return registry.get_asset_by_name(registry_name)
    found_usd_path = usd_path
    # Tagged 'object' unlike the generic asset, because this one carries its own usd_path.
    merged_tags = list(dict.fromkeys(["object", *SimReadyUsdObject.tags, *tags]))

    class SearchedSimReadyObject(SimReadyUsdObject):
        """A SimReady asset the search found for one object in a prompt."""

        name = registry_name
        tags = merged_tags

        def __init__(self, **kwargs: Any):
            kwargs.setdefault("usd_path", found_usd_path)
            super().__init__(**kwargs)

    registry.register(SearchedSimReadyObject, registry_name)
    return SearchedSimReadyObject
