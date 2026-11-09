# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural USD topology operations that cannot be expressed by composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def asset_prim(stage, path: str):
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


def apply_step(stage, step: Mapping[str, Any]) -> None:
    """Apply one procedural topology operation to a USD stage.

    Args:
        stage: Editable stage whose default prim contains the asset.
        step: Operation name and operation-specific selectors and parameters.
    """
    from isaaclab_arena.utils.usd.fixed_articulations import create_fixed_articulations
    from isaaclab_arena.utils.usd.mesh_partition import inspect_mesh_components, partition_mesh

    assert isinstance(step, Mapping), "Derivation step must be a mapping"
    operation = step["operation"]
    if operation == "partition_mesh":
        if expected := step.get("expected_geometry_sha256"):
            report = inspect_mesh_components(stage, step["prim"])
            assert report["geometry_sha256"] == expected, f"Mesh geometry changed: {step['prim']}"
        partition_mesh(stage, step["prim"], step["groups"], remainder=step.get("remainder", "Remaining"))
    elif operation == "fixed_articulations":
        create_fixed_articulations(stage, step["groups"])
    else:
        raise AssertionError(f"Unknown asset derivation operation: {operation}")
