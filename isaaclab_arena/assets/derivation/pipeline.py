# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Derive USD assets using procedural topology operations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


def derive_asset(package_root: Path, recipe: Mapping[str, Any]) -> None:
    """Apply procedural topology steps and save a flattened USD.

    Args:
        package_root: Staged package containing the original USD and its dependencies.
        recipe: Source/output entries, variants, topology steps, and required prim paths.
    """
    from isaaclab_arena.assets.derivation.operations import apply_step

    stage, output = _create_wrapper(package_root, recipe)
    for step in recipe.get("steps", []):
        apply_step(stage, step)
    _save_and_validate(stage, output, recipe, package_root)


def _package_path(package_root: Path, entry: str) -> Path:
    path = PurePosixPath(entry)
    assert entry and not path.is_absolute() and ".." not in path.parts, "Entry must stay inside the package"
    return package_root.joinpath(*path.parts)


def _create_wrapper(package_root: Path, recipe: Mapping[str, Any]):
    from pxr import Sdf, Usd, UsdGeom

    source = _package_path(package_root, recipe["source_entry"])
    output = _package_path(package_root, recipe["output_entry"])
    assert source != output, "Derived output must differ from the source entry"
    output.parent.mkdir(parents=True, exist_ok=True)
    source_stage = Usd.Stage.Open(str(source))
    assert source_stage and source_stage.GetDefaultPrim(), f"Cannot open source USD: {source}"
    root_path = Sdf.Path(recipe.get("root_prim", "/Asset"))
    assert (
        root_path.IsAbsolutePath() and root_path.IsPrimPath() and root_path.GetParentPath() == Sdf.Path.absoluteRootPath
    ), "root_prim must name a root-level prim"
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(source, output.parent), source_stage.GetDefaultPrim().GetPath())
    variants = root.GetVariantSets()
    for name, selection in recipe.get("variants", {}).items():
        if name in variants.GetNames() and selection in variants.GetVariantSet(name).GetVariantNames():
            variants.GetVariantSet(name).SetVariantSelection(selection)
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()
    return stage, output


def _save_and_validate(stage, output: Path, recipe: Mapping[str, Any], package_root: Path) -> None:
    from pxr import Usd, UsdUtils

    from isaaclab_arena.assets.derivation.operations import asset_prim

    layer = stage.Flatten()

    def make_package_relative(asset_path: str) -> str:
        path = Path(asset_path)
        if not path.is_absolute():
            return asset_path
        try:
            path.relative_to(package_root)
        except ValueError:
            return asset_path
        return os.path.relpath(path, output.parent)

    UsdUtils.ModifyAssetPaths(layer, make_package_relative)
    assert layer.Export(str(output)), f"Cannot export flattened derived USD: {output}"
    reopened = Usd.Stage.Open(str(output))
    assert reopened and reopened.GetDefaultPrim(), f"Cannot reopen derived USD: {output}"
    assert not reopened.GetCompositionErrors(), reopened.GetCompositionErrors()
    for path in recipe.get("required_prims", []):
        asset_prim(reopened, path)
