# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that review thumbnails use each native rigid asset configuration."""

import numpy as np
from copy import deepcopy
from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_native_preview_preserves_selected_geometry_and_sources(simulation_app, tmp_path: Path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.asset_usd import (
        cached_rigid_asset_preview,
    )

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Source").GetPrim()
    source_stage.SetDefaultPrim(source_root)
    UsdPhysics.RigidBodyAPI.Apply(source_root)
    shape_variants = source_root.GetVariantSets().AddVariantSet("shape")
    for variant_name, size in (("small", 0.2), ("large", 0.8)):
        shape_variants.AddVariant(variant_name)
        shape_variants.SetVariantSelection(variant_name)
        with shape_variants.GetVariantEditContext():
            cube = UsdGeom.Cube.Define(source_stage, "/Source/Cube")
            cube.CreateSizeAttr(size)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    shape_variants.SetVariantSelection("small")
    source_stage.GetRootLayer().Save()
    original_source = source_path.read_bytes()
    original_layer = source_stage.GetRootLayer().ExportToString()

    spawn_configs = [
        UsdFileCfg(usd_path=str(source_path), variants={"shape": "small"}),
        UsdFileCfg(usd_path=str(source_path), variants={"shape": "large"}),
        UsdFileCfg(usd_path=str(source_path), variants={"shape": "large"}, scale=(2.0, 3.0, 4.0)),
    ]
    expected_dimensions = ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8), (1.6, 2.4, 3.2))
    cache_dir = tmp_path / "thumbnails"
    preview_paths = []
    for spawn_cfg, dimensions in zip(spawn_configs, expected_dimensions, strict=True):
        original_settings = deepcopy(spawn_cfg.to_dict())
        preview_path = Path(cached_rigid_asset_preview(spawn_cfg, cache_dir))
        preview_paths.append(preview_path)
        assert preview_path.parent == cache_dir
        preview_stage = Usd.Stage.Open(str(preview_path))
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            preview_stage.GetDefaultPrim()
        )
        np.testing.assert_allclose(bounds.ComputeAlignedRange().GetSize(), dimensions, atol=1e-6)
        assert spawn_cfg.to_dict() == original_settings
        modification_time = preview_path.stat().st_mtime_ns
        assert cached_rigid_asset_preview(deepcopy(spawn_cfg), cache_dir) == str(preview_path)
        assert preview_path.stat().st_mtime_ns == modification_time

    assert len(set(preview_paths)) == len(spawn_configs)
    assert source_path.read_bytes() == original_source
    assert source_stage.GetRootLayer().ExportToString() == original_layer
    assert shape_variants.GetVariantSelection() == "small"
    return True


def test_native_preview_preserves_selected_geometry_and_sources(tmp_path: Path):
    assert run_function_with_persistent_simulation_app(
        _test_native_preview_preserves_selected_geometry_and_sources, tmp_path=tmp_path
    )
