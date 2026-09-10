# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export prepared layouts as replayable environment graph specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_events import get_pose_from_layout
from isaaclab_arena.relations.relations import IsAnchor

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.placement_result import PlacementResult


def settled_scene_spec(
    spec: ArenaEnvGraphSpec, layout: PlacementResult, assets_by_node_id: Mapping[str, Asset]
) -> ArenaEnvGraphSpec:
    """Return a graph with exact prepared clutter poses replacing placement relations.

    Args:
        spec: Original graph used to build the simulated scene.
        layout: Prepared layout whose captured objects passed the settle check.
        assets_by_node_id: Exact mapping returned when building this graph.

    Returns:
        Independent graph preserving assets and tasks, with fixed pose reset events.
    """
    assert layout.is_prepared, f"Only prepared clutter layouts can be exported: {layout.validation_results.report()}"
    assert not spec.object_sets, "Export object sets as concrete assets before saving a fixed scene"
    result = spec.model_copy(deep=True)
    nodes = {node.id: node for node in [*result.objects, result.background, result.embodiment]}
    node_ids_by_asset = {asset: node_id for node_id, asset in assets_by_node_id.items()}
    assert len(node_ids_by_asset) == len(assets_by_node_id), "Each graph node must map to a distinct asset"
    references = {ref.id for ref in result.object_references or []}
    frozen_ids = set()
    for asset in layout.positions:
        assert asset in node_ids_by_asset, f"Placement asset {asset.name!r} is missing from the graph mapping"
        node_id = node_ids_by_asset[asset]
        if node_id in references:
            assert asset.has_relation(IsAnchor), "Only anchored object references can be exported"
            continue
        assert node_id in nodes, f"Graph node {node_id!r} cannot store an initial pose"
        node = nodes[node_id]
        pose = asset.get_initial_pose() if asset.has_relation(IsAnchor) else get_pose_from_layout(asset, layout)
        assert pose is not None, f"Anchor '{asset.name}' has no pose"
        node.params["initial_pose"] = {
            "position_xyz": list(pose.position_xyz),
            "rotation_xyzw": list(pose.rotation_xyzw),
        }
        frozen_ids.add(node.id)
    # Every placed subject must be covered; retaining a relation could re-solve or re-pour the saved scene.
    assert all(
        r.subject in frozen_ids or r.subject in references for r in result.relations
    ), "Layout does not cover every relation subject"
    result.relations = []
    result.cli_override_specs = None
    result.placement_validators = None
    result.validate()
    return result
