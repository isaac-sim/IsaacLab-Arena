# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for source-neutral USD acquisition."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from isaaclab_arena.assets.sources.base import UsdSource
from isaaclab_arena.assets.sources.lightwheel import LightwheelUsdSource


def test_object_resolves_usd_source_path(tmp_path):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage

    stage = new_stage()
    add_body(stage, "body")
    usd_path = str(tmp_path / "item.usda")
    assert stage.Export(usd_path)

    source = Mock(spec=UsdSource)
    source.resolve.return_value = usd_path
    spawn_cfg_addon = {"variants": {"Quality": "High"}}

    obj = Object(
        name="item",
        usd_path=source,
        tags=["object"],
        spawn_cfg_addon=spawn_cfg_addon,
    )

    assert obj.usd_path == usd_path
    assert obj.object_type == ObjectType.RIGID
    assert obj.spawn_cfg_addon["variants"] == {"Quality": "High"}
    assert obj.tags == ["object"]
    assert spawn_cfg_addon == {"variants": {"Quality": "High"}}


def test_lightwheel_source_is_lazy_and_memoizes_acquisition():
    loader = SimpleNamespace(
        acquire_by_registry=Mock(return_value=("/cache/broccoli.usd", "broccoli", {})),
        client=SimpleNamespace(base_timeout=10),
    )
    loader_module = ModuleType("lightwheel_sdk.loader")
    loader_module.object_loader = loader
    package = ModuleType("lightwheel_sdk")
    package.loader = loader_module
    source = LightwheelUsdSource(registry_type="objects", registry_name=["broccoli"], file_type="USD")

    assert loader.acquire_by_registry.call_count == 0
    with patch.dict(
        "sys.modules",
        {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module},
    ):
        first = source.resolve()
        second = source.resolve()

    assert first is second
    assert first == "/cache/broccoli.usd"
    loader.acquire_by_registry.assert_called_once_with(
        registry_type="objects",
        registry_name=["broccoli"],
        file_type="USD",
    )


def test_library_objects_reuse_their_class_level_lightwheel_source(tmp_path):
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage

    stage = new_stage()
    add_body(stage, "body")
    usd_path = str(tmp_path / "asset.usda")
    assert stage.Export(usd_path)

    loader = SimpleNamespace(
        acquire_by_registry=Mock(return_value=(usd_path, "asset", {})),
        client=SimpleNamespace(base_timeout=10),
    )
    loader_module = ModuleType("lightwheel_sdk.loader")
    loader_module.object_loader = loader
    package = ModuleType("lightwheel_sdk")
    package.loader = loader_module
    source = LightwheelUsdSource(registry_type="objects", file_name="Asset001", file_type="USD")

    class SourcedObject(LibraryObject):
        registry_name = "sourced_object"
        tags = ["object"]
        usd_path = source

    with patch.dict(
        "sys.modules",
        {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module},
    ):
        first = SourcedObject(name="first")
        second = SourcedObject(name="second")

    assert first.usd_path == second.usd_path == usd_path
    loader.acquire_by_registry.assert_called_once()


def test_extract_asset_usd_path_resolves_source_without_constructing_asset():
    from isaaclab_arena.environment_spec.arena_env_graph_types import _extract_asset_usd_path

    source = Mock(spec=UsdSource)
    source.resolve.return_value = "/cache/asset.usd"

    class SourcedAsset:
        usd_path = source

        def __init__(self):
            raise AssertionError("Extracting the source path must not construct the asset")

    assert _extract_asset_usd_path(SourcedAsset) == "/cache/asset.usd"
    source.resolve.assert_called_once_with()
