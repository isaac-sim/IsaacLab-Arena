# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry
from isaaclab_arena.environment_spec.arena_env_graph_task_conversion_utils import build_task_from_spec
from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import has_light, open_stage

_DEFAULT_LIGHT_ASSET_NAME = "light"
_DEFAULT_LIGHT_NODE_ID = "auto_dome_light"
_DIRECTIONAL_LIGHT_ASSET_NAME = "directional_light"
_DIRECTIONAL_LIGHT_NODE_ID = "auto_directional_light"

if TYPE_CHECKING:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec


def build_arena_env_from_graph_spec(graph_spec: ArenaEnvGraphSpec, enable_cameras: bool = False) -> Any:
    """Build an IsaacLabArenaEnvironment from a validated ArenaEnvGraphSpec.

    Args:
        graph_spec: A validated graph spec (asset refs exist, ids unique, etc.).
        enable_cameras: Forwarded to the embodiment so its cameras are added.
    """
    # Lazy import to avoid pxr early import causing unit test failures.
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    assets_by_node_id = instantiate_assets_from_spec(graph_spec, AssetRegistry(), enable_cameras=enable_cameras)
    _ensure_scene_lighting(graph_spec, assets_by_node_id)
    _attach_spatial_relations_to_assets(graph_spec.relations, assets_by_node_id)
    scene_assets = [asset for node_id, asset in assets_by_node_id.items() if node_id != graph_spec.embodiment.id]
    return IsaacLabArenaEnvironment(
        name=graph_spec.env_name,
        scene=Scene(assets=scene_assets),
        embodiment=assets_by_node_id[graph_spec.embodiment.id],
        task=build_task_from_spec(graph_spec.task, assets_by_node_id),
        placer_params=build_checks_for_placer_params(graph_spec),
    )


def build_checks_for_placer_params(graph_spec: ArenaEnvGraphSpec) -> ObjectPlacerParams:
    """Build placement params defining what checks to run during layout validation for this env."""
    placement = graph_spec.placement_validators
    enabled = placement.enabled_checks if placement is not None else None
    required = placement.required_checks if placement is not None else None
    # TODO(xinjieyao): enable auto-identification of the placer params from env relations

    return ObjectPlacerParams(
        enabled_checks=set(enabled) if enabled is not None else None,
        required_checks=set(required) if required is not None else None,
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
    )


def _ensure_scene_lighting(graph_spec: ArenaEnvGraphSpec, assets_by_node_id: dict[str, Any]) -> None:
    """Ensure the scene has ambient lighting plus a directional light for lighting variations.

    A dome light is injected when the scene would otherwise render black (as before). A
    directional light (registry ``directional_light``) is then added on top so directional-lighting
    variations (``directional_light.direction.*`` / ``directional_light.intensity.*``) have a target;
    it is additive to whatever ambient or baked-in lighting the scene already has. Directional
    injection is skipped when the spec sets ``inject_directional_light: false`` (e.g. to match a
    reference env with no directional light).
    """
    if not _scene_already_has_light(graph_spec, assets_by_node_id):
        dome_node_id = _unique_node_id(set(assets_by_node_id), _DEFAULT_LIGHT_NODE_ID)
        light_cls = AssetRegistry().get_asset_by_name(_DEFAULT_LIGHT_ASSET_NAME)
        light_kwargs: dict[str, Any] = {}
        if graph_spec.dome_light_intensity is not None:
            # Deep-copy the class default so we don't mutate the shared spawner cfg every dome uses.
            spawner_cfg = copy.deepcopy(light_cls.default_spawner_cfg)
            spawner_cfg.intensity = graph_spec.dome_light_intensity
            light_kwargs["spawner_cfg"] = spawner_cfg
        assets_by_node_id[dome_node_id] = light_cls(**light_kwargs)
        intensity_note = (
            "" if graph_spec.dome_light_intensity is None else f" at intensity {graph_spec.dome_light_intensity}"
        )
        print(
            "INFO: no light found in scene or background USD(s); injected default light"
            f" '{dome_node_id}'{intensity_note}."
        )

    if graph_spec.inject_directional_light and not _scene_has_asset_named(
        assets_by_node_id, _DIRECTIONAL_LIGHT_ASSET_NAME
    ):
        directional_node_id = _unique_node_id(set(assets_by_node_id), _DIRECTIONAL_LIGHT_NODE_ID)
        assets_by_node_id[directional_node_id] = AssetRegistry().get_asset_by_name(_DIRECTIONAL_LIGHT_ASSET_NAME)()
        print(f"INFO: injected directional light '{directional_node_id}' for directional-lighting variations.")


def _scene_has_asset_named(assets_by_node_id: dict[str, Any], asset_name: str) -> bool:
    """Return whether any instantiated asset has the given registry name."""
    return any(getattr(asset, "name", None) == asset_name for asset in assets_by_node_id.values())


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


def _prim_path_for_relative(registry_name: str, prim_path: str) -> str:
    """Expand a relative prim suffix to the Isaac Lab runtime prim path."""
    if prim_path.startswith("{ENV_REGEX_NS}/"):
        return prim_path
    return f"{{ENV_REGEX_NS}}/{registry_name}/{prim_path.lstrip('/')}"


def instantiate_assets_from_spec(
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
        ref_params = dict(ref.params)
        openable_joint_name = ref_params.pop("openable_joint_name", None)
        prim_path = _prim_path_for_relative(graph_spec.background.registry_name, ref.prim_path)
        common_kwargs = {
            "name": ref.id,
            "prim_path": prim_path,
            "parent_asset": assets_by_node_id[ref.parent_id],
            "object_type": ref.object_type,
            **ref_params,
        }
        if openable_joint_name is not None:
            assets_by_node_id[ref.id] = OpenableObjectReference(
                openable_joint_name=openable_joint_name,
                **common_kwargs,
            )
        else:
            assets_by_node_id[ref.id] = ObjectReference(**common_kwargs)

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
