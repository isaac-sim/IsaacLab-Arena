# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic SimReady USD object registered for agent-generated environments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.simready_constants import (
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_SEARCH_REGISTRY_PREFIX,
    SIMREADY_USD_ARTICULATION_REGISTRY_NAME,
    SIMREADY_USD_BACKGROUND_REGISTRY_NAME,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
    SIMREADY_USD_OPENABLE_ARTICULATION_REGISTRY_NAME,
    SIMREADY_USD_PRESSABLE_ARTICULATION_REGISTRY_NAME,
)
from isaaclab_arena.utils.pose import Pose

_NON_IDENTIFIER_CHARACTERS = re.compile(r"[^a-z0-9]+")


@register_asset
class SimReadyUsdBackground(Background):
    """Spawn a SimReady background from an explicit USD path."""

    registry_name = SIMREADY_USD_BACKGROUND_REGISTRY_NAME
    tags = ["background", "sim-ready"]

    def __init__(
        self,
        usd_path: str,
        object_min_z: float,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs: Any,
    ):
        """Create a SimReady background from an explicit USD path."""
        assert usd_path, "simready_usd_background requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            download_central_asset_package,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = download_central_asset_package(usd_path)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        default_variants = {**SIMREADY_PHYSICS_VARIANTS, "PhysX": "Enabled"}
        spawn_cfg_addon.setdefault("variants", default_variants)
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            usd_path=usd_path,
            object_min_z=object_min_z,
            initial_pose=initial_pose,
            tags=self.tags,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


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
        **kwargs: Any,
    ):
        """Create an object from a SimReady USD file.

        Args:
            usd_path: USD path or URL of the SimReady asset.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            scale: Scale of the object. Defaults to ``(1.0, 1.0, 1.0)``.
            **kwargs: Extra arguments passed to ``Object``.
        """
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        assert usd_path, "simready_usd_object requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            download_and_wrap_central_asset,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = download_and_wrap_central_asset(usd_path)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        spawn_cfg_addon.setdefault("variants", dict(SIMREADY_PHYSICS_VARIANTS))
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


@register_asset
class SimReadyUsdArticulation(Object):
    """Spawn an articulated SimReady USD with optional initial-joint configuration."""

    registry_name = SIMREADY_USD_ARTICULATION_REGISTRY_NAME
    tags = ["object", "sim-ready"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        usd_path: str,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        initial_joint_positions: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        """Create an articulation from a SimReady USD path."""
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        assert usd_path, "simready_usd_articulation requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            download_central_asset_package,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = download_central_asset_package(usd_path)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        default_variants = {**SIMREADY_PHYSICS_VARIANTS, "PhysX": "Enabled"}
        spawn_cfg_addon.setdefault("variants", default_variants)
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)


@register_asset
class SimReadyUsdOpenableArticulation(SimReadyUsdArticulation, Openable):
    """Spawn an articulated SimReady USD that exposes one openable joint."""

    registry_name = SIMREADY_USD_OPENABLE_ARTICULATION_REGISTRY_NAME

    def __init__(self, openable_joint_name: str, **kwargs: Any):
        """Create an openable articulation from a SimReady USD path."""
        super().__init__(openable_joint_name=openable_joint_name, **kwargs)


@register_asset
class SimReadyUsdPressableArticulation(SimReadyUsdArticulation, Pressable):
    """Spawn an articulated SimReady USD that exposes one pressable joint."""

    registry_name = SIMREADY_USD_PRESSABLE_ARTICULATION_REGISTRY_NAME

    def __init__(self, pressable_joint_name: str, **kwargs: Any):
        """Create a pressable articulation from a SimReady USD path."""
        super().__init__(pressable_joint_name=pressable_joint_name, **kwargs)


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
    found_usd_path = usd_path
    # Tagged 'object' unlike the generic asset, because this one carries its own usd_path.
    merged_tags = list(dict.fromkeys(["object", *SimReadyUsdObject.tags, *tags]))

    class SearchedSimReadyObject(SimReadyUsdObject):
        """A SimReady asset the search found for one object in a prompt."""

        registry_name = searched_registry_name
        tags = merged_tags

        def __init__(self, **kwargs: Any):
            kwargs.setdefault("usd_path", found_usd_path)
            super().__init__(**kwargs)

    registry.register(SearchedSimReadyObject, searched_registry_name)
    return SearchedSimReadyObject
