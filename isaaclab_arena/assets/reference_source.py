# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pxr import Sdf, Usd

from isaaclab_arena.assets.object import Object

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_reference import ObjectReference


@dataclass(frozen=True)
class ReferenceSource:
    """Identify the spawned USD owner and the runtime subtree a reference may select."""

    asset: Object
    """Object that spawns the USD, including its variants and scale."""

    scope_prim_path: str
    """Runtime path of the parent object or referenced articulation."""

    @classmethod
    def from_parent_asset(cls, parent_asset: Object | ObjectReference) -> ReferenceSource:
        """Find the spawned owner while retaining the immediate parent's subtree."""
        asset = parent_asset if isinstance(parent_asset, Object) else parent_asset.root_asset
        return cls(asset=asset, scope_prim_path=parent_asset.get_prim_path())

    def prim_path_in_stage(self, runtime_prim_path: str, stage: Usd.Stage) -> str:
        """Map a runtime path within the scope to the source stage's default prim.

        Args:
            runtime_prim_path: Runtime path using the same environment prefix as the source asset.
            stage: Stage opened from the source asset's USD, including a spawned inspection stage.

        Returns:
            Absolute prim path in stage; this may differ from the path in the source USD file.
        """
        scope_path = self.scope_prim_path.rstrip("/")
        assert runtime_prim_path == scope_path or runtime_prim_path.startswith(
            scope_path + "/"
        ), f"Reference path '{runtime_prim_path}' must be beneath parent path '{scope_path}'"
        source_path = self.asset.get_prim_path().rstrip("/")
        assert scope_path == source_path or scope_path.startswith(
            source_path + "/"
        ), f"Reference scope '{scope_path}' must be beneath source path '{source_path}'"
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Parent USD must have a default prim"
        return str(default_prim.GetPath()) + runtime_prim_path.removeprefix(source_path)

    @staticmethod
    def prim_path_in_source_usd(prim_path_in_stage: str, stage: Usd.Stage) -> str:
        """Map a spawned-stage path to the USD file referenced by its default prim.

        Args:
            prim_path_in_stage: Absolute path under the spawned stage's default prim.
            stage: Inspection stage returned by Object.open_spawned_usd_stage().

        Returns:
            Path in the source USD file, for source geometry selection and collision exclusions.
        """
        default_prim = stage.GetDefaultPrim()
        # Nested references in the file must not replace the root layer's reference to the source file.
        [reference] = [
            arc
            for arc in Usd.PrimCompositionQuery.GetDirectReferences(default_prim).GetCompositionArcs()
            if arc.IsIntroducedInRootLayerStack()
        ]
        return (
            Sdf.Path(prim_path_in_stage).ReplacePrefix(default_prim.GetPath(), reference.GetTargetPrimPath()).pathString
        )
