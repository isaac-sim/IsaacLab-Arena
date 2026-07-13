# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generic SimReady USD object asset."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec


def test_simready_usd_object_registered():
    ensure_assets_registered()
    cls = AssetRegistry().get_asset_by_name("simready_usd_object")
    assert cls is SimReadyUsdObject
    assert cls.name == "simready_usd_object"
    assert "sim-ready" in cls.tags


def test_asset_spec_accepts_simready_usd_path():
    ensure_assets_registered()
    spec = AssetSpec(
        id="hammer",
        registry_name="simready_usd_object",
        params={"usd_path": "https://example.com/hammer.usd", "tags": ["sim-ready", "hammer"]},
    )
    assert spec.registry_name == "simready_usd_object"
    assert spec.params["usd_path"] == "https://example.com/hammer.usd"
