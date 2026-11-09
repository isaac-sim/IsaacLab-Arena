# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic SimReady USD object registered for agent-generated environments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.sources.base import UsdPathSource, UsdSource
from isaaclab_arena.assets.sources.simready import (
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_SEARCH_REGISTRY_PREFIX,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
    SimReadyCentralUsdSource,
    is_central_asset_url,
)
from isaaclab_arena.utils.pose import Pose

_NON_IDENTIFIER_CHARACTERS = re.compile(r"[^a-z0-9]+")


@register_asset
class SimReadyUsdObject(Object):
    """Spawn a SimReady asset from an explicit USD path supplied in graph params."""

    registry_name = SIMREADY_USD_OBJECT_REGISTRY_NAME

    tags = ["sim-ready"]

    object_type = ObjectType.RIGID
    """The search only accepts an asset with exactly one rigid body."""

    def __init__(
        self,
        usd_path: str,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        spawn_cfg_addon: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Create an object from a SimReady USD file.

        Args:
            usd_path: USD path or URL of the SimReady asset.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            scale: Scale of the object. Defaults to ``(1.0, 1.0, 1.0)``.
            spawn_cfg_addon: Additional USD spawn configuration.
            **kwargs: Extra arguments passed to ``Object``.
        """
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        assert usd_path, "simready_usd_object requires params.usd_path"
        spawn_cfg_addon = dict(spawn_cfg_addon or {})
        spawn_cfg_addon["variants"] = _merge_simready_variant_selections(spawn_cfg_addon.get("variants"))
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=_simready_usd_source(usd_path),
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


def simready_search_registry_name(search_phrase: str, usd_path: str) -> str:
    """Return a stable catalogue name for a phrase and asset path."""
    slug = _NON_IDENTIFIER_CHARACTERS.sub("_", search_phrase.strip().lower()).strip("_")
    assert slug, f"search phrase {search_phrase!r} has no characters usable in a registry name"
    digest = hashlib.sha256(usd_path.encode()).hexdigest()[:16]
    return f"{SIMREADY_SEARCH_REGISTRY_PREFIX}{slug}_{digest}"


def register_searched_simready_object(
    search_phrase: str,
    usd_path: str,
    tags: Sequence[str] = (),
) -> type[SimReadyUsdObject]:
    """Register a searched SimReady asset so it can be picked like any other catalogue object.

    Args:
        search_phrase: Phrase the asset was found for; used with its path to name the catalogue entry.
        usd_path: USD path or URL of the found asset.
        tags: Extra catalogue tags to expose alongside the SimReady ones.

    Returns:
        The registered asset class.
    """
    searched_registry_name = simready_search_registry_name(search_phrase, usd_path)
    registry = AssetRegistry()
    if registry.is_registered(searched_registry_name, ensure_loaded=False):
        return registry.get_asset_by_name(searched_registry_name)
    # Tagged 'object' unlike the generic asset, because this one carries its own usd_path.
    merged_tags = list(dict.fromkeys(["object", *SimReadyUsdObject.tags, *tags]))

    class SearchedSimReadyObject(SimReadyUsdObject):
        """A SimReady asset the search found for one object in a prompt."""

        registry_name = searched_registry_name
        tags = merged_tags

        def __init__(self, **kwargs: Any):
            kwargs.setdefault("usd_path", usd_path)
            super().__init__(**kwargs)

    registry.register(SearchedSimReadyObject, searched_registry_name)
    return SearchedSimReadyObject


def _simready_usd_source(usd_path: str) -> UsdSource:
    """Build a SimReady source for a direct path or Central URL."""
    if is_central_asset_url(usd_path):
        return SimReadyCentralUsdSource(usd_path)
    return UsdPathSource(usd_path)


def _merge_simready_variant_selections(overrides: object | None) -> dict[str, str]:
    """Merge SimReady selections with dictionary or configclass overrides."""
    if overrides is None:
        override_values = {}
    elif isinstance(overrides, Mapping):
        override_values = dict(overrides)
    else:
        to_dict = getattr(overrides, "to_dict", None)
        assert callable(to_dict), f"variants must be a mapping or configclass instance, got {type(overrides)}"
        override_values = to_dict()
        assert isinstance(override_values, dict), "configclass variants must convert to a dictionary"
    assert all(
        isinstance(key, str) and isinstance(value, str) for key, value in override_values.items()
    ), "variant names and selections must be strings"
    return {**SIMREADY_PHYSICS_VARIANTS, **override_values}
