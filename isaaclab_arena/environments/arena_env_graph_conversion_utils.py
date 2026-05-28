# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_task_conversion_utils import build_task_or_sequence
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphSpatialConstraintType,
    ArenaEnvGraphStateSpec,
)
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec


def build_arena_env_from_graph_spec(spec: ArenaEnvGraphSpec) -> Any:
    """Create an IsaacLabArenaEnvironment from an already-validated graph spec."""
    # TODO(xinjieyao, 2026-05-26): aggregate every state_spec into a single combined initial state instead of
    # picking one. For now we just take the first state_spec, which is the initial state
    # for the first task — this matches the previous default behavior.
    state_spec = spec.state_specs[0] if spec.state_specs else None

    assets_by_id = _instantiate_node_assets(spec.nodes, AssetRegistry())
    if state_spec is not None:
        _apply_state_spatial_constraints(state_spec, assets_by_id)

    embodiment = None
    scene_assets: list[Any] = []
    # TODO(xinjieyao, 2026-05-26): include lighting later
    for node in spec.nodes:
        if node.type == ArenaEnvGraphNodeType.EMBODIMENT:
            assert embodiment is None, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
            embodiment = assets_by_id[node.id]
        elif node.type in (ArenaEnvGraphNodeType.OBJECT, ArenaEnvGraphNodeType.OBJECT_REFERENCE):
            scene_assets.append(assets_by_id[node.id])

    return IsaacLabArenaEnvironment(
        name=spec.env_name,
        scene=Scene(assets=scene_assets),
        embodiment=embodiment,
        task=build_task_or_sequence(spec.tasks, assets_by_id),
    )


def _instantiate_node_assets(nodes: list[ArenaEnvGraphNodeSpec], asset_registry: Any) -> dict[str, Any]:
    """Create concrete asset entities for graph nodes and wire object-reference nodes.

    Upstream contract:
      * Nodes are ordered so an OBJECT_REFERENCE appears after its parent — a single pass is
        enough; the parent lookup in `assets_by_id` would `KeyError` otherwise.
      * `node.params` is emitted in the asset constructor's exact kwarg form, including any
        `instance_name` needed to disambiguate duplicate assets.
    """
    assets_by_id: dict[str, Any] = {}
    for node in nodes:
        if node.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            assert isinstance(node, ArenaEnvGraphObjectReferenceNodeSpec)
            assets_by_id[node.id] = ObjectReference(
                name=node.name,
                prim_path=node.prim_path,
                parent_asset=assets_by_id[node.parent],
                object_type=node.object_type,
                **node.params,
            )
            continue
        asset_cls = asset_registry.get_asset_by_name(node.name)
        assets_by_id[node.id] = asset_cls(**node.params)
    return assets_by_id


def _apply_state_spatial_constraints(state_spec: ArenaEnvGraphStateSpec, assets_by_id: dict[str, Any]) -> None:
    """Attach relation entities or fixed poses to the environment's initial state."""
    for constraint in state_spec.spatial_constraints:
        parent_asset = assets_by_id[constraint.parent]
        # Note(xinjieyao, 2026-05-26): at_pose constraint is a special case that is handled differently.
        # Ideally it shall be handled within the placer module.
        if constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
            params = dict(constraint.params)
            position_xyz = params.pop("position_xyz", None)
            rotation_xyzw = params.pop("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))
            assert position_xyz is not None, f"at_pose constraint '{constraint.id}' requires params.position_xyz"
            assert not params, f"Unsupported at_pose params for constraint '{constraint.id}': {sorted(params)}"
            parent_asset.set_initial_pose(Pose(position_xyz=position_xyz, rotation_xyzw=rotation_xyzw))
            continue

        relation_cls = relation_class_for_spatial_constraint_type(constraint.type)
        assert relation_cls is not None, f"Unsupported spatial constraint type '{constraint.type.value}'"
        # unary relation: parent_asset is the only endpoint
        # non-unary relation: parent_asset and child_asset are both endpoints
        if relation_cls.is_unary():
            parent_asset.add_relation(relation_cls(**constraint.params))
        else:
            assets_by_id[constraint.child].add_relation(relation_cls(parent_asset, **constraint.params))
