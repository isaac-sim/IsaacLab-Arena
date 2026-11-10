# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Author computed structural edits in a USDA layer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def author_structural_layer(source: Path, output: Path, recipe: Mapping[str, Any]) -> None:
    """Author a structural layer that continues to reference its source.

    Args:
        source: Retrieved source USD entrypoint.
        output: USDA structural layer to create.
        recipe: Variant selections, structural steps, and required prim paths.
    """
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

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
    root.GetReferences().AddReference(str(source), source_stage.GetDefaultPrim().GetPath())
    apply_usd_variant_selections(stage, recipe.get("variants", {}))
    assert not stage.GetCompositionErrors(), stage.GetCompositionErrors()

    for step in recipe["steps"]:
        _apply_structural_step(stage, step)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output))
    assert reopened and reopened.GetDefaultPrim(), f"Cannot reopen structural layer: {output}"
    assert not reopened.GetCompositionErrors(), reopened.GetCompositionErrors()
    for path in recipe.get("required_prims", []):
        _get_asset_prim(reopened, path)


def _get_asset_prim(stage, path: str):
    """Return an existing prim addressed relative to the asset's default prim."""
    from pxr import Sdf

    root = stage.GetDefaultPrim().GetPath()
    target = root if path == "." else Sdf.Path(path)
    if not target.IsAbsolutePath():
        target = root.AppendPath(target)
    assert target.IsPrimPath() and target.HasPrefix(root), f"Prim must stay inside the asset: {path}"
    prim = stage.GetPrimAtPath(target)
    assert prim, f"Missing asset prim: {path}"
    return prim


def _apply_structural_step(stage, step: Mapping[str, Any]) -> None:
    """Apply one computed structural operation to a USD stage."""
    from isaaclab_arena_environments.workbench_cleanup.asset_processing.fixed_articulations import (
        create_fixed_articulation,
    )
    from isaaclab_arena_environments.workbench_cleanup.asset_processing.mesh_partition import (
        inspect_mesh_components,
        partition_mesh,
    )

    assert isinstance(step, Mapping), "Structural step must be a mapping"
    operation = step["operation"]
    if operation == "partition_mesh":
        if expected := step.get("expected_geometry_sha256"):
            report = inspect_mesh_components(stage, step["prim"])
            assert report["geometry_sha256"] == expected, f"Mesh geometry changed: {step['prim']}"
        partition_mesh(stage, step["prim"], step["groups"], remainder=step.get("remainder", "Remaining"))
    elif operation == "fixed_articulation":
        create_fixed_articulation(stage, step["name"], step["frame_mesh"], step["joints"])
    else:
        raise AssertionError(f"Unknown structural operation: {operation}")
