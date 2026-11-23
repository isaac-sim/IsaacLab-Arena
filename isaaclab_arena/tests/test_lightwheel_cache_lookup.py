# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

import pytest

from isaaclab_arena.assets.sources import lightwheel


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "lightwheel_sdk"
    root.mkdir()
    monkeypatch.setattr(lightwheel, "lightwheel_cache_root", lambda: root)
    return root


def make_object(root, dir_name, marker="id1.txt", with_usd=True):
    obj_dir = root / "object" / dir_name
    obj_dir.mkdir(parents=True)
    if with_usd:
        (obj_dir / f"{dir_name}.usd").write_text("usd")
    if marker is not None:
        (obj_dir / marker).write_text("")


def make_floorplan(root, name, version="uuid1", register=True):
    floorplan_dir = root / "floorplan"
    (floorplan_dir / name).mkdir(parents=True)
    (floorplan_dir / name / "scene.usd").write_text("usd")
    if register:
        (floorplan_dir / "usd_version.json").write_text(json.dumps({name: version}))


def get_cached_object_usd_path(file_name):
    return lightwheel.LightwheelObjectUsdSource(
        registry_type="objects", file_name=file_name, file_type="USD"
    )._get_cached_usd_path()


def get_cached_floorplan_usd_path(scene, layout_id, style_id, backend):
    return lightwheel.LightwheelFloorplanUsdSource(scene, layout_id, style_id, backend)._get_cached_usd_path()


def test_get_cached_object_usd_path_returns_complete_asset(cache_root):
    make_object(cache_root, "Microwave039")
    expected = str(cache_root / "object" / "Microwave039" / "Microwave039.usd")
    assert get_cached_object_usd_path("Microwave039") == expected


def test_get_cached_object_usd_path_matches_file_name_case_insensitively(cache_root):
    make_object(cache_root, "Beer016")
    expected = str(cache_root / "object" / "Beer016" / "Beer016.usd")
    assert get_cached_object_usd_path("beer016") == expected


def test_get_cached_object_usd_path_misses_without_version_marker(cache_root):
    make_object(cache_root, "Microwave039", marker=None)
    assert get_cached_object_usd_path("Microwave039") is None


def test_get_cached_object_usd_path_misses_when_usd_missing(cache_root):
    make_object(cache_root, "Microwave039", with_usd=False)
    assert get_cached_object_usd_path("Microwave039") is None


def test_get_cached_object_usd_path_misses_in_empty_cache(cache_root):
    assert get_cached_object_usd_path("Microwave039") is None


def test_get_cached_floorplan_usd_path_returns_complete_asset(cache_root):
    make_floorplan(cache_root, "robocasa-robocasakitchen-7-10")
    expected = str(cache_root / "floorplan" / "robocasa-robocasakitchen-7-10" / "scene.usd")
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") == expected


def test_get_cached_floorplan_usd_path_hits_when_version_entry_is_null(cache_root):
    make_floorplan(cache_root, "robocasa-robocasakitchen-7-10", version=None)
    expected = str(cache_root / "floorplan" / "robocasa-robocasakitchen-7-10" / "scene.usd")
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") == expected


def test_get_cached_floorplan_usd_path_misses_without_version_entry(cache_root):
    make_floorplan(cache_root, "robocasa-robocasakitchen-7-10", register=False)
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") is None


def test_get_cached_floorplan_usd_path_misses_on_corrupt_version_file(cache_root):
    make_floorplan(cache_root, "robocasa-robocasakitchen-7-10")
    (cache_root / "floorplan" / "usd_version.json").write_text("{not json")
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") is None


def test_get_cached_floorplan_usd_path_misses_without_version_file(cache_root):
    make_floorplan(cache_root, "robocasa-robocasakitchen-7-10", register=False)
    assert not (cache_root / "floorplan" / "usd_version.json").exists()
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") is None


def test_floorplan_cache_dir_name_matches_installed_sdk(cache_root):
    floorplan_module = pytest.importorskip("lightwheel_sdk.loader.floorplan")
    loader_classes = [
        cls for _, cls in inspect.getmembers(floorplan_module, inspect.isclass) if "_usd_cache_name" in vars(cls)
    ]
    assert loader_classes, "no loader class defines _usd_cache_name"
    sdk_name = loader_classes[0]._usd_cache_name(None, "robocasa", "robocasakitchen", 7, 10)
    make_floorplan(cache_root, sdk_name)
    expected = str(cache_root / "floorplan" / sdk_name / "scene.usd")
    assert get_cached_floorplan_usd_path("robocasakitchen", 7, 10, "robocasa") == expected
