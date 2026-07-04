# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The opt-in Maple staging asset override must not leak into the registered (shared) asset class."""

import json
from types import SimpleNamespace

from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    _PROD_NUCLEUS_HOST,
    _STAGING_NUCLEUS_HOST,
    _apply_local_droid_asset_override,
    _load_local_asset_provenance,
    _local_asset_path,
    _local_asset_subclass,
    _staging_subclass,
    _to_staging_url,
)


def test_pick_targets_cli_is_fail_closed():
    """--pick_targets uses nargs='+' / default None: absent -> single-object; a present flag must carry names."""
    import argparse

    import pytest

    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironment

    parser = argparse.ArgumentParser()
    PickAndPlaceMapleTableEnvironment.add_cli_args(parser)

    # Absent -> None (stock single-object path, unchanged).
    assert parser.parse_args([]).pick_targets is None
    # Present with names -> ordered list.
    assert parser.parse_args(["--pick_targets", "a", "b", "c"]).pick_targets == ["a", "b", "c"]
    assert parser.parse_args(["--pick_targets", "a", "b", "c", "d", "e"]).pick_targets == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]
    # Present but empty is rejected (nargs='+'), so it cannot silently fall back to single-object.
    with pytest.raises(SystemExit):
        parser.parse_args(["--pick_targets"])


def test_droid_stand_staging_override_is_instance_local():
    """Overriding one DROID embodiment's stand to staging must NOT leak into a fresh (stock) embodiment.

    The stand AssetBaseCfg is a class-level configclass default; the override must deep-copy it, not mutate
    the shared object in place. (Deferred imports: instantiating the embodiment needs the booted app.)
    """
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import _apply_staging_stand_override

    staged_emb = DroidAbsoluteJointPositionEmbodiment(enable_cameras=False)
    prod_url = staged_emb.scene_config.stand.spawn.usd_path
    assert _PROD_NUCLEUS_HOST in prod_url

    staged_url = _apply_staging_stand_override(staged_emb)
    assert _STAGING_NUCLEUS_HOST in staged_url
    assert staged_emb.scene_config.stand.spawn.usd_path == staged_url

    # A fresh stock embodiment built afterward must still point at production — no leak via the shared default.
    stock_emb = DroidAbsoluteJointPositionEmbodiment(enable_cameras=False)
    assert stock_emb.scene_config.stand.spawn.usd_path == prod_url, "staging override leaked into a stock embodiment"
    assert _PROD_NUCLEUS_HOST in stock_emb.scene_config.stand.spawn.usd_path


def test_to_staging_url_swaps_host_only():
    prod = f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/x/maple_table.usda"
    staged = _to_staging_url(prod)
    assert _STAGING_NUCLEUS_HOST in staged and _PROD_NUCLEUS_HOST not in staged
    # Only the host changes; the asset path is preserved exactly.
    assert staged.split(_STAGING_NUCLEUS_HOST, 1)[1] == prod.split(_PROD_NUCLEUS_HOST, 1)[1]


def test_staging_subclass_does_not_mutate_registered_class():
    """_staging_subclass returns a subclass with the staging URL; the input class is left untouched."""

    class _FakeAsset:
        usd_path = f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/x/maple_table.usda"

    original = _FakeAsset.usd_path
    staged_cls, staged_url = _staging_subclass(_FakeAsset)

    # The subclass points at staging...
    assert staged_cls is not _FakeAsset
    assert issubclass(staged_cls, _FakeAsset)
    assert staged_cls.usd_path == staged_url
    assert _STAGING_NUCLEUS_HOST in staged_url
    # ...but the registered (shared) class is unchanged — no leak into later non-staging jobs/rebuilds.
    assert _FakeAsset.usd_path == original
    assert _PROD_NUCLEUS_HOST in _FakeAsset.usd_path


def test_local_asset_path_is_host_qualified_and_fail_closed(tmp_path):
    import pytest

    source = f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/example.usd"
    expected = tmp_path / _PROD_NUCLEUS_HOST / "Assets/Isaac/6.0/example.usd"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"usd")

    assert _local_asset_path(source, tmp_path) == str(expected)
    with pytest.raises(FileNotFoundError, match="baked CAP asset is missing"):
        _local_asset_path(f"https://{_PROD_NUCLEUS_HOST}/Assets/Isaac/6.0/missing.usd", tmp_path)
    with pytest.raises(RuntimeError, match="unsupported CAP asset source URL"):
        _local_asset_path("https://example.com/Assets/asset.usd", tmp_path)


def test_local_asset_subclass_does_not_mutate_registered_class(tmp_path):
    source = f"https://{_STAGING_NUCLEUS_HOST}/Assets/Isaac/6.0/example.usda"
    expected = tmp_path / _STAGING_NUCLEUS_HOST / "Assets/Isaac/6.0/example.usda"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"usd")

    class _FakeAsset:
        usd_path = source

    localized_cls, source_url, local_path = _local_asset_subclass(_FakeAsset, tmp_path)
    assert localized_cls is not _FakeAsset
    assert localized_cls.usd_path == str(expected)
    assert source_url == source
    assert local_path == str(expected)
    assert _FakeAsset.usd_path == source


def test_local_droid_override_deep_copies_both_spawn_configs(tmp_path):
    robot_url = f"https://{_PROD_NUCLEUS_HOST}/Assets/robot.usd"
    stand_url = f"https://{_STAGING_NUCLEUS_HOST}/Assets/stand.usda"
    for url in (robot_url, stand_url):
        parsed_host, parsed_path = url.removeprefix("https://").split("/", 1)
        path = tmp_path / parsed_host / parsed_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"usd")

    robot = SimpleNamespace(spawn=SimpleNamespace(usd_path=robot_url))
    stand = SimpleNamespace(spawn=SimpleNamespace(usd_path=stand_url))
    embodiment = SimpleNamespace(scene_config=SimpleNamespace(robot=robot, stand=stand))
    paths = _apply_local_droid_asset_override(embodiment, tmp_path)

    assert embodiment.scene_config.robot is not robot
    assert embodiment.scene_config.stand is not stand
    assert robot.spawn.usd_path == robot_url
    assert stand.spawn.usd_path == stand_url
    assert paths["robot_source_usd"] == robot_url
    assert paths["stand_source_usd"] == stand_url
    assert paths["robot_local_usd"] == embodiment.scene_config.robot.spawn.usd_path
    assert paths["stand_local_usd"] == embodiment.scene_config.stand.spawn.usd_path


def test_local_asset_provenance_requires_schema_and_tree_hash(tmp_path, monkeypatch):
    import pytest

    provenance_path = tmp_path / "CAP_ASSET_PROVENANCE.json"
    provenance_path.write_text(json.dumps({"schema_version": 1, "tree_sha256": "a" * 64}), encoding="utf-8")
    assert _load_local_asset_provenance(tmp_path)["tree_sha256"] == "a" * 64

    monkeypatch.setenv("CAP_IMAGE_ASSET_TREE_SHA256", "b" * 64)
    with pytest.raises(RuntimeError, match="does not match image pin"):
        _load_local_asset_provenance(tmp_path)
    monkeypatch.delenv("CAP_IMAGE_ASSET_TREE_SHA256")

    provenance_path.write_text(json.dumps({"schema_version": 1, "tree_sha256": "nope"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid baked CAP asset provenance"):
        _load_local_asset_provenance(tmp_path)
