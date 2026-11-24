# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for source-neutral USD acquisition."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from isaaclab_arena.assets.sources.base import UsdSource
from isaaclab_arena.assets.sources.lightwheel import (
    LightwheelFloorplanUsdSource,
    LightwheelObjectUsdSource,
    LightwheelSource,
)

BLOCK_LIGHTWHEEL_SDK = {"lightwheel_sdk": None, "lightwheel_sdk.loader": None}
"""sys.modules entries that make ``import lightwheel_sdk`` raise ImportError."""


@pytest.fixture(autouse=True)
def empty_lightwheel_cache(tmp_path, monkeypatch):
    """Point the Lightwheel cache at an empty directory so tests never see the developer's real cache."""
    cache_root = tmp_path / "empty_lightwheel_cache"
    cache_root.mkdir()
    monkeypatch.setattr("isaaclab_arena.assets.sources.lightwheel.lightwheel_cache_root", lambda: cache_root)
    return cache_root


def _make_loader_modules(**loaders):
    loader_module = ModuleType("lightwheel_sdk.loader")
    for name, loader in loaders.items():
        setattr(loader_module, name, loader)
    package = ModuleType("lightwheel_sdk")
    package.loader = loader_module
    return {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module}


def test_object_acquires_usd_source_path(tmp_path):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage

    stage = new_stage()
    add_body(stage, "body")
    usd_path = str(tmp_path / "item.usda")
    assert stage.Export(usd_path)

    source = Mock(spec=UsdSource)
    source.to_local_usd_path.return_value = usd_path
    spawn_cfg_addon = {"variants": {"Quality": "High"}}

    obj = Object(
        name="item",
        usd_source=source,
        tags=["object"],
        spawn_cfg_addon=spawn_cfg_addon,
    )

    assert obj.usd_path == usd_path
    assert obj.usd_source is source
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
    source = LightwheelObjectUsdSource(
        registry_type="objects", file_name="Broccoli015", file_type="USD", registry_name=["broccoli"]
    )

    assert loader.acquire_by_registry.call_count == 0
    with patch.dict(
        "sys.modules",
        {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module},
    ):
        first = source.to_local_usd_path()
        second = source.to_local_usd_path()

    assert first is second
    assert first == "/cache/broccoli.usd"
    loader.acquire_by_registry.assert_called_once_with(
        registry_type="objects",
        file_name="Broccoli015",
        file_type="USD",
        registry_name=["broccoli"],
    )


def test_lightwheel_floorplan_source_is_lazy_and_memoizes_acquisition():
    loader = SimpleNamespace(
        get_usd=Mock(return_value=("/cache/kitchen.usd", {})),
        client=SimpleNamespace(base_timeout=10),
    )
    loader_module = ModuleType("lightwheel_sdk.loader")
    loader_module.floorplan_loader = loader
    package = ModuleType("lightwheel_sdk")
    package.loader = loader_module
    source = LightwheelFloorplanUsdSource(
        scene="robocasakitchen",
        layout_id=6,
        style_id=10,
        backend="robocasa",
    )

    assert loader.get_usd.call_count == 0
    with patch.dict(
        "sys.modules",
        {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module},
    ):
        first = source.to_local_usd_path()
        second = source.to_local_usd_path()

    assert first is second
    assert first == "/cache/kitchen.usd"
    loader.get_usd.assert_called_once_with(
        scene="robocasakitchen",
        layout_id=6,
        style_id=10,
        backend="robocasa",
    )


def test_lightwheel_object_source_returns_cached_path_without_importing_sdk(empty_lightwheel_cache, capsys):
    object_dir = empty_lightwheel_cache / "object" / "Microwave039"
    object_dir.mkdir(parents=True)
    (object_dir / "Microwave039.usd").write_text("")
    (object_dir / "version.txt").write_text("1")
    source = LightwheelObjectUsdSource(registry_type="fixtures", file_name="Microwave039", file_type="USD")

    with patch.dict("sys.modules", BLOCK_LIGHTWHEEL_SDK):
        assert source.to_local_usd_path() == str(object_dir / "Microwave039.usd")
    assert "Lightwheel cache hit: lightwheel-object/microwave039" in capsys.readouterr().out


def test_lightwheel_object_source_calls_sdk_on_cache_miss(capsys):
    loader = SimpleNamespace(
        acquire_by_registry=Mock(return_value=("/cache/microwave.usd", "microwave", {})),
        client=SimpleNamespace(base_timeout=10),
    )
    source = LightwheelObjectUsdSource(registry_type="fixtures", file_name="Microwave039", file_type="USD")

    with patch.dict("sys.modules", _make_loader_modules(object_loader=loader)):
        assert source.to_local_usd_path() == "/cache/microwave.usd"
    assert "Lightwheel cache miss: lightwheel-object/microwave039" in capsys.readouterr().out

    loader.acquire_by_registry.assert_called_once_with(
        registry_type="fixtures", file_name="Microwave039", file_type="USD"
    )


def test_lightwheel_object_source_requires_a_file_name():
    with pytest.raises(AssertionError, match="needs a file_name"):
        LightwheelObjectUsdSource(registry_type="objects", file_name="", file_type="USD")


def test_lightwheel_floorplan_source_returns_cached_path_without_importing_sdk(empty_lightwheel_cache):
    floorplan_root = empty_lightwheel_cache / "floorplan"
    scene_dir = floorplan_root / "robocasa-robocasakitchen-6-10"
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene.usd").write_text("")
    (floorplan_root / "usd_version.json").write_text('{"robocasa-robocasakitchen-6-10": "1"}')
    source = LightwheelFloorplanUsdSource(scene="robocasakitchen", layout_id=6, style_id=10, backend="robocasa")

    with patch.dict("sys.modules", BLOCK_LIGHTWHEEL_SDK):
        assert source.to_local_usd_path() == str(scene_dir / "scene.usd")


@pytest.mark.parametrize("file_path", [None, ""])
@pytest.mark.parametrize("kind", ["object", "floorplan"])
def test_lightwheel_source_rejects_missing_path_without_caching_it(kind, file_path):
    valid_path = Path("/cache/asset.usd")
    acquire = Mock(side_effect=[(file_path, "asset", {}), (valid_path, "asset", {})])
    loader_module = ModuleType("lightwheel_sdk.loader")
    if kind == "object":
        loader_module.object_loader = SimpleNamespace(acquire_by_registry=acquire)
        source = LightwheelObjectUsdSource(registry_type="objects", file_name="Asset001", file_type="USD")
    else:
        loader_module.floorplan_loader = SimpleNamespace(get_usd=acquire)
        source = LightwheelFloorplanUsdSource(scene="robocasakitchen", layout_id=6, style_id=10, backend="robocasa")
    package = ModuleType("lightwheel_sdk")
    package.loader = loader_module

    with patch.dict("sys.modules", {"lightwheel_sdk": package, "lightwheel_sdk.loader": loader_module}):
        with pytest.raises(AssertionError, match="Lightwheel did not return"):
            source.to_local_usd_path()
        assert source.to_local_usd_path() == str(valid_path)
        assert source.to_local_usd_path() == str(valid_path)
    assert acquire.call_count == 2


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
    source = LightwheelObjectUsdSource(registry_type="objects", file_name="Asset001", file_type="USD")

    class SourcedObject(LibraryObject):
        registry_name = "sourced_object"
        tags = ["object"]
        usd_source = source

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
    source.to_local_usd_path.return_value = "/cache/asset.usd"

    class SourcedAsset:
        usd_source = source

        def __init__(self):
            raise AssertionError("Extracting the source path must not construct the asset")

    assert _extract_asset_usd_path(SourcedAsset) == "/cache/asset.usd"
    source.to_local_usd_path.assert_called_once_with()


def test_lightwheel_source_cannot_be_instantiated_without_the_four_hooks():
    with pytest.raises(TypeError):
        LightwheelSource()
