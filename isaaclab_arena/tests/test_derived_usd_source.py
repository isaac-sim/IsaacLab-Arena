# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deterministic derived USD package caching."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from isaaclab_arena.assets.sources.base import UsdSource
from isaaclab_arena.assets.sources.derived import DerivedUsdSource

MODULE = "isaaclab_arena.assets.derivation.cache"


class _CountingSource(UsdSource):
    def __init__(self, identity: str, path: Path):
        self._identity = identity
        self.path = path
        self.resolve_calls = 0

    def identity(self) -> str:
        return self._identity

    def resolve(self) -> str:
        self.resolve_calls += 1
        return str(self.path)


def _write_recipe(tmp_path: Path, source: _CountingSource, **overrides: Any) -> Path:
    recipe = {
        "source_identity": source.identity(),
        "source_entry": "simready_usd/item.usda",
        "output_entry": "derived/item.usda",
        "expected_source_sha256": hashlib.sha256(source.path.read_bytes()).hexdigest(),
        "steps": [],
        **overrides,
    }
    recipe_path = tmp_path / f"recipe-{len(list(tmp_path.glob('recipe-*.json')))}.json"
    recipe_path.write_text(json.dumps(recipe))
    return recipe_path


@pytest.fixture
def source(tmp_path: Path) -> _CountingSource:
    package = tmp_path / "source-package"
    entry = package / "simready_usd/item.usda"
    entry.parent.mkdir(parents=True)
    entry.write_text('#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {}\n')
    (package / "textures").mkdir()
    (package / "textures/color.png").write_bytes(b"texture")
    return _CountingSource("test:package", entry)


def test_derived_source_builds_complete_package_and_reuses_cache(tmp_path, source, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    recipe = _write_recipe(tmp_path, source)

    first_path = Path(DerivedUsdSource(source, str(recipe)).resolve())

    assert first_path.is_file()
    assert (first_path.parents[1] / "textures/color.png").read_bytes() == b"texture"
    assert source.resolve_calls == 1
    manifest = json.loads((first_path.parents[1] / "manifest.json").read_text())
    assert manifest["source_identity"] == source.identity()
    assert manifest["output_entry"] == "derived/item.usda"

    cache_only_source = _CountingSource(source.identity(), Path("/missing/source.usd"))
    second_path = Path(DerivedUsdSource(cache_only_source, str(recipe)).resolve())

    assert second_path == first_path
    assert cache_only_source.resolve_calls == 0


def test_derived_source_rebuilds_corrupt_output(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    recipe = _write_recipe(tmp_path, source)
    derived = DerivedUsdSource(source, str(recipe))
    output = Path(derived.resolve())
    output.write_text("corrupt")

    rebuilt_source = _CountingSource(source.identity(), source.path)
    rebuilt = Path(DerivedUsdSource(rebuilt_source, str(recipe)).resolve())

    assert "corrupt" not in rebuilt.read_text()
    assert rebuilt_source.resolve_calls == 1


def test_derived_source_accepts_direct_path_and_reuses_cache_without_original(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    source_path = str(source.path)
    recipe = _write_recipe(tmp_path, source, source_identity=source_path)

    first_path = DerivedUsdSource(source_path, str(recipe)).resolve()
    source.path.unlink()

    assert DerivedUsdSource(source_path, str(recipe)).resolve() == first_path
    assert Path(first_path).is_file()


def test_derived_source_accepts_resolve_only_source_with_explicit_identity(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    recipe = _write_recipe(tmp_path, source)

    class ResolveOnlySource(UsdSource):
        def resolve(self) -> str:
            return source.resolve()

    with pytest.raises(AssertionError, match="source_identity"):
        DerivedUsdSource(ResolveOnlySource(), str(recipe))
    assert source.resolve_calls == 0

    first = DerivedUsdSource(ResolveOnlySource(), str(recipe), source_identity=source.identity())
    first_path = first.resolve()
    second = DerivedUsdSource(ResolveOnlySource(), str(recipe), source_identity=source.identity())

    assert second.resolve() == first_path
    assert source.resolve_calls == 1


def test_derived_source_recipe_and_pipeline_version_change_cache_key(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    first_recipe = _write_recipe(tmp_path, source)
    first_path = DerivedUsdSource(source, str(first_recipe)).resolve()

    changed_recipe = _write_recipe(tmp_path, source, output_entry="derived/renamed.usda")
    changed_path = DerivedUsdSource(source, str(changed_recipe)).resolve()
    assert changed_path != first_path

    monkeypatch.setattr(f"{MODULE}.DERIVATION_PIPELINE_VERSION", 2)
    versioned_path = DerivedUsdSource(source, str(first_recipe)).resolve()
    assert versioned_path != first_path


def test_derived_source_rejects_wrong_source_hash(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    recipe = _write_recipe(tmp_path, source, expected_source_sha256="0" * 64)

    with pytest.raises(AssertionError, match="does not match recipe hash"):
        DerivedUsdSource(source, str(recipe)).resolve()


def test_derived_source_rejects_wrong_source_identity(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    recipe = _write_recipe(tmp_path, source, source_identity="other:package")

    with pytest.raises(AssertionError, match="does not match"):
        DerivedUsdSource(source, str(recipe)).resolve()
    assert source.resolve_calls == 0


def test_derived_source_removes_failed_staging_directory(tmp_path, source, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    recipe = _write_recipe(
        tmp_path,
        source,
        steps=[{"operation": "missing"}],
    )

    with pytest.raises(AssertionError, match="Unknown asset derivation operation"):
        DerivedUsdSource(source, str(recipe)).resolve()

    derived_root = cache / "derived"
    assert not [path for path in derived_root.iterdir() if path.is_dir()]
    assert not list(derived_root.glob("*/manifest.json"))


def test_derived_source_serializes_concurrent_builds(tmp_path, source, monkeypatch):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    recipe = _write_recipe(tmp_path, source)
    derived_sources = [DerivedUsdSource(source, str(recipe)) for _ in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda derived: derived.resolve(), derived_sources))

    assert len(set(paths)) == 1
    assert source.resolve_calls == 1


def test_derived_source_preserves_symlinked_snapshot_package_path(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: cache)
    blob = tmp_path / "blobs/source"
    blob.parent.mkdir()
    blob.write_text('#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {}\n')
    snapshot_entry = tmp_path / "snapshots/revision/asset/simready_usd/item.usda"
    snapshot_entry.parent.mkdir(parents=True)
    snapshot_entry.symlink_to(blob)
    source = _CountingSource("test:symlink-package", snapshot_entry)
    recipe = _write_recipe(
        tmp_path,
        source,
        source_entry="simready_usd/item.usda",
    )

    output = Path(DerivedUsdSource(source, str(recipe)).resolve())

    assert output.is_file()
