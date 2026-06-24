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
from isaaclab_arena.environments.graph_spec_utils import relation_class_for_spatial_constraint_type, unique_node_id
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import has_light, open_stage

# Registered asset name (DomeLight) used as the default light when a scene has none.
_DEFAULT_LIGHT_ASSET_NAME = "light"

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec


def build_arena_env_from_graph_spec(graph_spec: ArenaEnvGraphSpec, enable_cameras: bool = False) -> Any:
    """Build an IsaacLabArenaEnvironment from a validated ``ArenaEnvGraphSpec``.

    Args:
        graph_spec: A validated graph spec (node refs exist, ids unique, etc.).
        enable_cameras: Forwarded to the embodiment node so its cameras are added.
    """
    # TODO(xinjieyao, 2026-05-26): aggregate every state_spec into a single combined initial state instead of
    # picking one. For now we just take the first state_spec, which is the initial state
    # for the first task — this matches the previous default behavior.
    initial_state_spec = graph_spec.state_specs[0] if graph_spec.state_specs else None

    # 1. Materialize every graph node into a live asset, keyed by node id so spatial
    #    constraints and task args can reference each node by its graph-local id.
    assets_by_node_id = _instantiate_assets_from_nodes(graph_spec.nodes, AssetRegistry(), enable_cameras=enable_cameras)

    # 2. Guarantee the scene has a light. A graph (or its background USD) with no light renders
    #    black, which today only surfaces once the policy runner launches the generated env. When
    #    nothing provides a light, inject a default DomeLight as a tracked LIGHTING node so the
    #    augmented graph stays faithful to what actually spawns.
    _ensure_scene_lighting(graph_spec, assets_by_node_id)

    # 3. Wire the initial state's spatial relations / fixed poses into those assets.
    if initial_state_spec is not None:
        _attach_spatial_constraints_to_assets(initial_state_spec, assets_by_node_id)

    # 4. Partition nodes into the env's embodiment (exactly one) and its scene assets.
    embodiment, scene_assets = _partition_nodes_into_embodiment_and_scene(graph_spec.nodes, assets_by_node_id)

    # 5. Resolve task specs against the same assets_by_node_id so task args bind to the
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
    builder substitutes a ``NoEmbodiment`` for scene-only specs. Every non-embodiment node is a
    scene asset; ``Scene.add_asset`` validates each instance is an ``Asset``/``RigidObjectSet`` and
    raises otherwise (embodiments are ``Asset`` too, so node type — not isinstance — separates them).
    """
    embodiment = None
    scene_assets: list[Asset] = []
    for node_spec in node_specs:
        if node_spec.type == ArenaEnvGraphNodeType.EMBODIMENT:
            assert embodiment is None, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
            embodiment = assets_by_node_id[node_spec.id]
        else:
            scene_assets.append(assets_by_node_id[node_spec.id])
    # No embodiment node -> embodiment stays None; the env builder resolves that to a
    # NoEmbodiment (`self.arena_env.embodiment or NoEmbodiment()`), so scene-only specs are valid.
    return embodiment, scene_assets


def _ensure_scene_lighting(graph_spec: ArenaEnvGraphSpec, assets_by_node_id: dict[str, Any]) -> None:
    """Inject a default light when the scene would otherwise render black.

    Mutates ``graph_spec.nodes`` and ``assets_by_node_id`` in place: appends a ``LIGHTING`` node
    backed by the registered ``DomeLight`` when no explicit light is declared and no base scene USD
    already ships one. Tracking it as a real node keeps the augmented graph faithful to what spawns.
    No-op when a light is already present, so a scene that defines or bakes in its own light is left
    untouched (avoids double-lighting).
    """
    if _scene_already_has_light(graph_spec, assets_by_node_id):
        return

    node_id = unique_node_id({node.id for node in graph_spec.nodes}, "auto_dome_light")
    light_node = ArenaEnvGraphNodeSpec(id=node_id, name=_DEFAULT_LIGHT_ASSET_NAME, type=ArenaEnvGraphNodeType.LIGHTING)
    graph_spec.nodes.append(light_node)
    assets_by_node_id[node_id] = AssetRegistry().get_asset_by_name(_DEFAULT_LIGHT_ASSET_NAME)()
    print(f"INFO: no light found in scene or background USD(s); injected default light node '{node_id}'.")


def _scene_already_has_light(graph_spec: ArenaEnvGraphSpec, assets_by_node_id: dict[str, Any]) -> bool:
    """Return whether the scene is already lit, either explicitly or via a baked-in USD light."""
    # Explicit: a LIGHTING node, or any materialized asset tagged as a light (e.g. DomeLight).
    if any(node.type == ArenaEnvGraphNodeType.LIGHTING for node in graph_spec.nodes):
        return True
    if any("light" in (getattr(asset, "tags", None) or []) for asset in assets_by_node_id.values()):
        return True
    # Implicit: any base scene asset whose USD already contains a light.
    for node in graph_spec.nodes:
        if node.type == ArenaEnvGraphNodeType.EMBODIMENT or node.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            continue
        asset = assets_by_node_id[node.id]
        # Only USD-backed base assets can carry a baked-in light; assets with an explicit
        # spawner_cfg (DomeLight, GroundPlane, primitives) declare their own prim instead.
        usd_path = getattr(asset, "usd_path", None)
        if usd_path is not None and getattr(asset, "spawner_cfg", None) is None:
            with open_stage(usd_path) as stage:
                if has_light(stage):
                    return True
    return False


def _instantiate_assets_from_nodes(
    node_specs: list[ArenaEnvGraphNodeSpec], asset_registry: Any, enable_cameras: bool = False
) -> dict[str, type[Asset]]:
    """Return ``{node.id: live_asset}`` after a single pass over ``node_specs``."""
    assets_by_node_id: dict[str, type[Asset]] = {}
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
            params = dict(node_spec.params)
            # Embodiment cameras are enabled thru the flag passed to the env builder.
            if node_spec.type == ArenaEnvGraphNodeType.EMBODIMENT and enable_cameras:
                params.setdefault("enable_cameras", True)
            assets_by_node_id[node_spec.id] = asset_class(**params)
    return assets_by_node_id


def _attach_spatial_constraints_to_assets(
    state_spec: ArenaEnvGraphStateSpec, assets_by_node_id: dict[str, type[Asset]]
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
