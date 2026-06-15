# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environments.arena_env_graph_task_conversion_utils import build_task_from_specs
from isaaclab_arena.environments.arena_env_graph_types import (
    ArenaEnvGraphNodeSpec,
    ArenaEnvGraphNodeType,
    ArenaEnvGraphObjectReferenceNodeSpec,
    ArenaEnvGraphStateSpec,
)
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec


def build_arena_env_from_graph_spec(graph_spec: ArenaEnvGraphSpec) -> Any:
    """Build an IsaacLabArenaEnvironment from a validated ``ArenaEnvGraphSpec``.

    Precondition: ``graph_spec`` is already validated (node refs exist, ids unique, etc.).
    """
    # TODO(xinjieyao, 2026-05-26): aggregate every state_spec into a single combined initial state instead of
    # picking one. For now we just take the first state_spec, which is the initial state
    # for the first task — this matches the previous default behavior.
    initial_state_spec = graph_spec.state_specs[0] if graph_spec.state_specs else None

    # 1. Materialize every graph node into a live asset, keyed by node id so spatial
    #    constraints and task args can reference each node by its graph-local id.
    assets_by_node_id = _instantiate_assets_from_nodes(graph_spec.nodes, AssetRegistry())

    # 2. Wire the initial state's spatial relations / fixed poses into those assets.
    if initial_state_spec is not None:
        _attach_spatial_constraints_to_assets(initial_state_spec, assets_by_node_id)

    # 3. Partition nodes into the env's embodiment (exactly one) and its scene assets.
    embodiment, scene_assets = _partition_nodes_into_embodiment_and_scene(graph_spec.nodes, assets_by_node_id)

    # 4. Resolve task specs against the same assets_by_node_id so task args bind to the
    #    actual asset instances created in step 1 (not duplicates).
    return IsaacLabArenaEnvironment(
        name=graph_spec.env_name,
        scene=Scene(assets=scene_assets),
        embodiment=embodiment,
        task=build_task_from_specs(graph_spec.tasks, assets_by_node_id),
    )


def _partition_nodes_into_embodiment_and_scene(
    node_specs: list[ArenaEnvGraphNodeSpec], assets_by_node_id: dict[str, Any]
) -> tuple[Any, list[Asset]]:
    """Split materialized nodes into the optional embodiment asset and the list of scene assets.

    Asserts at most one EMBODIMENT node; zero is allowed and returns ``None`` here — the env
    builder substitutes a ``NoEmbodiment`` for scene-only specs. BACKGROUND / OBJECT /
    OBJECT_REFERENCE nodes become scene assets; any other node type raises. Lighting is not handled yet.
    """
    embodiment = None
    scene_assets: list[Asset] = []
    # TODO(xinjieyao, 2026-05-26): include lighting later
    for node_spec in node_specs:
        if node_spec.type == ArenaEnvGraphNodeType.EMBODIMENT:
            assert embodiment is None, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
            embodiment = assets_by_node_id[node_spec.id]
        elif node_spec.type in (
            ArenaEnvGraphNodeType.BACKGROUND,
            ArenaEnvGraphNodeType.OBJECT,
            ArenaEnvGraphNodeType.OBJECT_REFERENCE,
        ):
            scene_assets.append(assets_by_node_id[node_spec.id])
        else:
            raise ValueError(f"Unsupported node type: {node_spec.type}")
    # No embodiment node -> embodiment stays None; the env builder resolves that to a
    # NoEmbodiment (`self.arena_env.embodiment or NoEmbodiment()`), so scene-only specs are valid.
    return embodiment, scene_assets


def _instantiate_assets_from_nodes(node_specs: list[ArenaEnvGraphNodeSpec], asset_registry: Any) -> dict[str, Any]:
    """Return ``{node.id: live_asset}`` after a single pass over ``node_specs``.

    Each ``node_spec.params`` is forwarded verbatim to the asset constructor. Assumes parent
    nodes precede their OBJECT_REFERENCE children — guaranteed by graph-spec reference validation.
    """
    assets_by_node_id: dict[str, Any] = {}
    for node_spec in node_specs:
        # OBJECT_REFERENCE wraps a USD prim inside an already-instantiated parent asset
        # (e.g. a table inside a kitchen background). Validation guarantees the parent
        # precedes the reference, so it is already in assets_by_node_id here.
        if node_spec.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            assert isinstance(node_spec, ArenaEnvGraphObjectReferenceNodeSpec)
            assets_by_node_id[node_spec.id] = ObjectReference(
                name=node_spec.name,
                prim_path=node_spec.prim_path,
                parent_asset=assets_by_node_id[node_spec.parent],
                object_type=node_spec.object_type,
                **node_spec.params,
            )
        else:
            # Standard nodes (object / background / embodiment): look up the registered class
            # by name and instantiate with the spec's verbatim kwargs.
            asset_class = asset_registry.get_asset_by_name(node_spec.name)
            assets_by_node_id[node_spec.id] = asset_class(**node_spec.params)
    return assets_by_node_id


def _attach_spatial_constraints_to_assets(
    state_spec: ArenaEnvGraphStateSpec, assets_by_node_id: dict[str, Any]
) -> None:
    """Attach one Relation per spatial constraint to the asset(s) it targets, in place."""
    for spatial_constraint in state_spec.spatial_constraints:
        subject_asset = assets_by_node_id[spatial_constraint.subject]
        relation_class = relation_class_for_spatial_constraint_type(spatial_constraint.kind)
        # Unary relations (IS_ANCHOR, POSITION_LIMITS, ...) attach to the subject asset.
        # Binary relations (ON, NEXT_TO, ...) attach to the subject; the reference node
        # is passed as the Relation constructor's first arg — matches how add_relation is wired.
        if relation_class.is_unary():
            subject_asset.add_relation(relation_class(**spatial_constraint.params))
            # An is_anchor asset must have a fixed initial pose for the placement solver.
            # If the asset class does not declare one, default to the world origin so
            # LLM-generated specs (which never set an explicit pose) work out of the box.
            if spatial_constraint.kind == "is_anchor" and subject_asset.get_initial_pose() is None:
                subject_asset.set_initial_pose(Pose.identity())
        else:
            reference_asset = assets_by_node_id[spatial_constraint.reference]
            subject_asset.add_relation(relation_class(reference_asset, **spatial_constraint.params))

        # TODO(qianl): add back support for ``at_pose``.
        # AT_POSE has no Relation class — it pins the parent's initial pose directly,
        # bypassing the solver. Need to be handled in the placer module.
