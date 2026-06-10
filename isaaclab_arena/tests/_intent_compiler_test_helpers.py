# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for test_asset_matcher.py and test_intent_compiler.py.

Leading underscore keeps pytest from collecting this module as a test file.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec, Item
from isaaclab_arena.agentic_environment_generation.intent_compiler import IntentCompiler
from isaaclab_arena.environments.arena_env_graph_types import SpatialRelationSpec, TaskSpec


@dataclass
class FakeAsset:
    """Minimal stand-in for the asset classes the resolvers inspect.

    Real asset classes are decorated classes pulled in via
    ``ensure_assets_registered()``.  The resolvers only ever read ``.name``
    and ``.tags``, so a plain dataclass keeps tests independent of Isaac Sim.
    """

    name: str
    tags: list[str]


class FakeAssetRegistry:
    """Duck-typed AssetRegistry for unit tests.

    Implements the methods the resolvers call — ``is_registered``,
    ``get_asset_by_name``, ``get_assets_by_tag``, ``get_assets_with_all_tags``,
    ``get_all_keys`` — without
    pulling in isaaclab.  We deliberately don't subclass :class:`AssetRegistry`
    because it uses ``SingletonMeta``, which would force test-isolation gymnastics.
    Duck-typing via the resolver's ``registry`` argument is the supported
    injection point.
    """

    def __init__(self, assets: list[FakeAsset]) -> None:
        self._by_name: dict[str, FakeAsset] = {a.name: a for a in assets}

    def is_registered(self, key: str) -> bool:
        return key in self._by_name

    def get_asset_by_name(self, name: str) -> FakeAsset:
        assert name in self._by_name, f"unregistered asset: {name}"
        return self._by_name[name]

    def get_assets_by_tag(self, tag: str) -> list[FakeAsset]:
        return [a for a in self._by_name.values() if tag in a.tags]

    def get_assets_with_all_tags(self, tags: list[str]) -> list[str]:
        return sorted(asset.name for asset in self._by_name.values() if all(tag in asset.tags for tag in tags))

    def get_all_keys(self) -> list[str]:
        return list(self._by_name)


def default_assets() -> list[FakeAsset]:
    """Small but representative catalog covering all three asset categories.

    Object names intentionally include the suffix conventions seen in the real
    registry (e.g. ``bowl_ycb_robolab``) so substring-match tests exercise
    realistic fuzzy/substring behaviour.
    """
    return [
        FakeAsset(name="maple_table", tags=["background"]),
        FakeAsset(name="franka_ik", tags=["embodiment", "ik"]),
        FakeAsset(name="franka_joint_pos", tags=["embodiment"]),
        FakeAsset(name="bowl_ycb_robolab", tags=["object", "bowl"]),
        FakeAsset(name="avocado01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="apple01_fruits_robolab", tags=["object", "fruit"]),
        FakeAsset(name="cracker_box", tags=["object", "graspable"]),
    ]


def make_compiler(assets: list[FakeAsset] | None = None) -> IntentCompiler:
    """Return an :class:`IntentCompiler` backed by a :class:`FakeAssetRegistry`."""
    return IntentCompiler(registry=FakeAssetRegistry(assets or default_assets()))


def make_scene(
    *,
    background: str = "maple_table",
    embodiment: str = "franka_ik",
    items: list[Item] | None = None,
    initial_state_graph: list[SpatialRelationSpec] | None = None,
    tasks: list[TaskSpec] | None = None,
) -> EnvironmentIntentSpec:
    """Build an :class:`EnvironmentIntentSpec` with sane defaults.

    Tests that don't care about a particular field can leave it unset; the
    defaults produce a valid spec that resolves cleanly against
    :func:`default_assets`.
    """
    return EnvironmentIntentSpec(
        reasoning="test scene",
        background=background,
        embodiment=embodiment,
        items=items or [],
        initial_state_graph=initial_state_graph or [],
        tasks=tasks
        or [
            TaskSpec(
                kind="PickAndPlaceTask",
                params={
                    "pick_up_object": "placeholder",
                    "destination_location": "placeholder",
                    "background_scene": "maple_table",
                },
                description="placeholder",
            )
        ],
    )
