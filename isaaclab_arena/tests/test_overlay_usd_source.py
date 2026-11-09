# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cached USDA overlays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from isaaclab_arena.assets.sources.base import UsdSource
from isaaclab_arena.assets.sources.overlay import OverlayUsdSource

MODULE = "isaaclab_arena.assets.derivation.cache"


class _CountingSource(UsdSource):
    def __init__(self, path: Path):
        self.path = path
        self.resolve_calls = 0

    def identity(self) -> str:
        return "test:overlay-source"

    def resolve(self) -> str:
        self.resolve_calls += 1
        return str(self.path)


@pytest.fixture
def source(tmp_path: Path) -> _CountingSource:
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Source").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Source/Body").GetPrim())
    stage.GetRootLayer().Save()
    return _CountingSource(path)


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    path = tmp_path / "overlay.usda"
    path.write_text("""#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @source.usda@</Source>
)
{
    over "Body" (
        prepend apiSchemas = ["PhysicsMassAPI"]
    )
    {
        float physics:mass = 2
    }
}
""")
    return path


def test_overlay_rewrites_reference_and_reuses_cache(tmp_path, monkeypatch, source, overlay):
    from pxr import Usd, UsdPhysics

    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    expected_hash = hashlib.sha256(source.path.read_bytes()).hexdigest()
    output = Path(OverlayUsdSource(source, str(overlay), expected_source_sha256=expected_hash).resolve())

    stage = Usd.Stage.Open(str(output))
    assert not stage.GetCompositionErrors()
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath("/Asset/Body")).GetMassAttr().Get() == 2
    assert source.resolve_calls == 1

    missing_source = _CountingSource(tmp_path / "missing.usda")
    assert OverlayUsdSource(missing_source, str(overlay), expected_source_sha256=expected_hash).resolve() == str(output)
    assert missing_source.resolve_calls == 0


def test_overlay_rejects_wrong_source_hash(tmp_path, monkeypatch, source, overlay):
    monkeypatch.setattr(f"{MODULE}.get_arena_asset_cache_dir", lambda: tmp_path / "cache")

    with pytest.raises(AssertionError, match="does not match expected hash"):
        OverlayUsdSource(source, str(overlay), expected_source_sha256="0" * 64).resolve()
