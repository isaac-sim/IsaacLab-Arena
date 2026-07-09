# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry
from isaaclab_arena.environment_spec.arena_env_graph_task_conversion_utils import build_task_from_spec
from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import has_light, open_stage

_DEFAULT_LIGHT_ASSET_NAME = "light"
_DEFAULT_LIGHT_NODE_ID = "auto_dome_light"

if TYPE_CHECKING:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec


def build_arena_env_from_graph_spec(graph_spec: ArenaEnvGraphSpec, enable_cameras: bool = False) -> Any:
    """Build an IsaacLabArenaEnvironment from a validated ArenaEnvGraphSpec.

    Args:
        graph_spec: A validated graph spec (asset refs exist, ids unique, etc.).
        enable_cameras: Forwarded to the embodiment so its cameras are added.
    """
    assets_by_node_id = _instantiate_assets_from_spec(graph_spec, AssetRegistry(), enable_cameras=enable_cameras)
    _ensure_scene_lighting(graph_spec, assets_by_node_id)
    _attach_spatial_relations_to_assets(graph_spec.relations, assets_by_node_id)
    scene_assets = [asset for node_id, asset in assets_by_node_id.items() if node_id != graph_spec.embodiment.id]
    return IsaacLabArenaEnvironment(
        name=graph_spec.env_name,
        scene=Scene(assets=scene_assets),
        embodiment=assets_by_node_id[graph_spec.embodiment.id],
        task=build_task_from_spec(graph_spec.task, assets_by_node_id),
    )


def _ensure_scene_lighting(graph_spec: ArenaEnvGraphSpec, assets_by_node_id: dict[str, Any]) -> None:
    """Inject a default light when the scene would otherwise render black."""
    if _scene_already_has_light(graph_spec, assets_by_node_id):
        return

    node_id = _unique_node_id(set(assets_by_node_id), _DEFAULT_LIGHT_NODE_ID)
    assets_by_node_id[node_id] = AssetRegistry().get_asset_by_name(_DEFAULT_LIGHT_ASSET_NAME)()
    print(f"INFO: no light found in scene or background USD(s); injected default light '{node_id}'.")


def _unique_node_id(existing_ids: set[str], base: str) -> str:
    """Return the first non-colliding id from ``base``, ``base_1``, ``base_2``, ... given ``existing_ids``."""
    if base not in existing_ids:
        return base
    suffix = 1
    while f"{base}_{suffix}" in existing_ids:
        suffix += 1
    return f"{base}_{suffix}"


def _scene_already_has_light(graph_spec: ArenaEnvGraphSpec, assets_by_node_id: dict[str, Any]) -> bool:
    """Return whether the scene is already lit, either explicitly or via a baked-in USD light."""
    if any("light" in (getattr(asset, "tags", None) or []) for asset in assets_by_node_id.values()):
        return True
    for asset_spec in [graph_spec.background, *graph_spec.objects]:
        asset = assets_by_node_id[asset_spec.id]
        usd_path = getattr(asset, "usd_path", None)
        if usd_path is not None and getattr(asset, "spawner_cfg", None) is None:
            with open_stage(usd_path) as stage:
                if has_light(stage):
                    return True
    return False


def _instantiate_assets_from_spec(
    graph_spec: ArenaEnvGraphSpec, asset_registry: Any, enable_cameras: bool = False
) -> dict[str, type[Asset]]:
    """Return ``{asset.id: live_asset}`` after materializing the typed graph spec."""
    assets_by_node_id: dict[str, type[Asset]] = {}

    embodiment_params = dict(graph_spec.embodiment.params)
    if enable_cameras:
        embodiment_params.setdefault("enable_cameras", True)
    assets_by_node_id[graph_spec.embodiment.id] = asset_registry.get_asset_by_name(graph_spec.embodiment.registry_name)(
        **embodiment_params
    )

    assets_by_node_id[graph_spec.background.id] = asset_registry.get_asset_by_name(graph_spec.background.registry_name)(
        **graph_spec.background.params
    )

    for obj in graph_spec.objects:
        params = dict(obj.params)
        params.setdefault("instance_name", obj.id)
        assets_by_node_id[obj.id] = asset_registry.get_asset_by_name(obj.registry_name)(**params)

    for ref in graph_spec.object_references or []:
        assert ref.prim_path is not None, "Object reference must have a prim path"
        assets_by_node_id[ref.id] = ObjectReference(
            name=ref.id,
            prim_path=ref.prim_path,
            parent_asset=assets_by_node_id[ref.parent_id],
            object_type=ref.object_type,
            **ref.params,
        )

    return assets_by_node_id


def _attach_spatial_relations_to_assets(
    relations: list[SpatialRelationSpec], assets_by_node_id: dict[str, type[Asset]]
) -> None:
    """Attach one Relation per spatial relation to the asset(s) it targets, in place."""
    for relation in relations:
        subject_asset = assets_by_node_id[relation.subject]
        relation_class = ObjectRelationLibraryRegistry().get_object_relation_by_name(relation.kind)
        if relation_class.is_unary():
            subject_asset.add_relation(relation_class(**relation.params))
            if relation.kind == "is_anchor" and subject_asset.get_initial_pose() is None:
                subject_asset.set_initial_pose(Pose.identity())
        else:
            reference_asset = assets_by_node_id[relation.reference]
            subject_asset.add_relation(relation_class(reference_asset, **relation.params))
