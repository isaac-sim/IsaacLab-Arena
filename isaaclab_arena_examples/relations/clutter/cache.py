# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-pose scene YAML serialization for offline clutter caches."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.utils.pose import Pose


def scene_with_cached_poses(
    spec: ArenaEnvGraphSpec, poses: Mapping[str, Pose], assets_by_node_id: Mapping[str, Asset]
) -> ArenaEnvGraphSpec:
    """Return an independent scene graph with cached rigid-object poses.

    Args:
        spec: Concrete input scene without placement relations or unresolved object sets.
        poses: Environment-local poses keyed by the rigid objects' scene keys.
        assets_by_node_id: Exact graph-node-to-asset mapping from scene construction.
    """
    assert not spec.relations, "A fixed-pose scene cannot retain placement relations"
    assert not spec.object_sets, "Resolve object sets to concrete assets before caching"
    result = spec.model_copy(deep=True)
    nodes = [result.background, result.embodiment, *result.objects]
    nodes_by_key = {assets_by_node_id[node.id].get_scene_key(): node for node in nodes}
    assert len(nodes_by_key) == len(nodes), "Graph nodes must map to distinct scene keys"
    assert set(poses) <= set(
        nodes_by_key
    ), f"Cannot serialize rigid objects missing from the graph: {set(poses) - set(nodes_by_key)}"
    for key, pose in poses.items():
        assert all(
            math.isfinite(value) for value in (*pose.position_xyz, *pose.rotation_xyzw)
        ), f"Cached pose for {key!r} must be finite"
        assert math.isclose(
            sum(value * value for value in pose.rotation_xyzw), 1.0, abs_tol=1e-4
        ), f"Cached rotation for {key!r} must be a unit quaternion"
        nodes_by_key[key].params["initial_pose"] = {
            "position_xyz": list(pose.position_xyz),
            "rotation_xyzw": list(pose.rotation_xyzw),
        }
    result.validate()
    return result


def write_scene_cache(spec: ArenaEnvGraphSpec, path: Path) -> None:
    """Publish a complete YAML file atomically, refusing to replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".clutter-") as directory:
        temporary = Path(directory) / path.name
        spec.write_yaml(temporary)
        # A same-filesystem hard link publishes the complete file without overwriting a raced writer.
        os.link(temporary, path)
