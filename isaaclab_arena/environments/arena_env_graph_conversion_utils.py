# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from isaaclab_arena.environments.arena_env_graph_task_conversion_utils import build_task_or_sequence
from isaaclab_arena.environments.graph_spec_utils import spatial_constraint_relation_classes

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_env_graph_spec import (
        ArenaEnvGraphNodeSpec,
        ArenaEnvGraphSpatialConstraintSpec,
        ArenaEnvGraphSpec,
        ArenaEnvGraphStateSpec,
        ArenaEnvGraphTaskSpec,
    )


def build_arena_env_from_graph_spec(
    spec: ArenaEnvGraphSpec,
    state_spec_id: str | None = None,
) -> Any:
    """Create an IsaacLabArenaEnvironment from an already-validated graph spec."""
    state_spec = _select_state_spec(spec, spec.tasks, state_spec_id)
    if state_spec is not None:
        _validate_initial_state_supported_for_conversion(state_spec)

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    assets_by_id = _instantiate_node_assets(spec.nodes, AssetRegistry())
    if state_spec is not None:
        _apply_state_spatial_constraints(state_spec, assets_by_id)

    return IsaacLabArenaEnvironment(
        name=spec.env_name,
        scene=Scene(
            assets=[assets_by_id[node.id] for node in spec.nodes if node.type != ArenaEnvGraphNodeType.EMBODIMENT]
        ),
        embodiment=_select_embodiment(spec.nodes, assets_by_id),
        task=build_task_or_sequence(spec.tasks, assets_by_id),
    )


def _validate_initial_state_supported_for_conversion(state_spec: ArenaEnvGraphStateSpec) -> None:
    """Reject startup-state constraints that the runtime converter cannot build yet, e.g. all relations defined in existing relations.py module."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType

    relation_classes = spatial_constraint_relation_classes()
    for constraint in state_spec.spatial_constraints:
        if constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
            continue
        assert (
            relation_classes.get(constraint.type) is not None
        ), f"Spatial constraint type '{constraint.type.value}' is not supported for initial state conversion"


def _select_state_spec(
    spec: ArenaEnvGraphSpec,
    task_specs: list[ArenaEnvGraphTaskSpec],
    state_spec_id: str | None,
) -> ArenaEnvGraphStateSpec | None:
    """Choose which graph state becomes the scene's initial layout."""
    if state_spec_id is None and task_specs:
        state_spec_id = task_specs[0].initial_state_spec_id
    if state_spec_id is None:
        if not spec.state_specs:
            return None
        assert len(spec.state_specs) == 1, "state_spec_id is required when multiple state specs exist"
        return spec.state_specs[0]
    state_spec = spec.state_specs_by_id.get(state_spec_id)
    assert state_spec is not None, f"Unknown state spec id '{state_spec_id}'"
    return state_spec


def _instantiate_node_assets(nodes: list[ArenaEnvGraphNodeSpec], asset_registry: Any) -> dict[str, Any]:
    """Create concrete asset entities for graph nodes and wire object-reference nodes."""
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.environments.arena_env_graph_spec import (
        ArenaEnvGraphNodeType,
        ArenaEnvGraphObjectReferenceNodeSpec,
    )

    assets_by_id: dict[str, Any] = {}
    for node in nodes:
        if node.type == ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            continue
        asset_cls = asset_registry.get_asset_by_name(node.name)
        kwargs = dict(node.params)
        if (
            node.id != node.name
            and "instance_name" not in kwargs
            and _explicitly_accepts_kwarg(asset_cls, "instance_name")
        ):
            kwargs["instance_name"] = node.id
        assets_by_id[node.id] = asset_cls(**kwargs)

    for node in nodes:
        if node.type != ArenaEnvGraphNodeType.OBJECT_REFERENCE:
            continue
        assert isinstance(node, ArenaEnvGraphObjectReferenceNodeSpec)
        params = dict(node.params)
        reserved_keys = {"name", "prim_path", "parent_asset", "object_type"}
        overlapping_keys = sorted(reserved_keys.intersection(params))
        assert (
            not overlapping_keys
        ), f"Object reference node '{node.id}' params override reserved fields: {overlapping_keys}"
        assets_by_id[node.id] = ObjectReference(
            name=node.name,
            prim_path=node.prim_path,
            parent_asset=assets_by_id[node.parent],
            object_type=node.object_type,
            **params,
        )
    return assets_by_id


def _explicitly_accepts_kwarg(callable_obj: Any, name: str) -> bool:
    """Check whether a constructor can accept a named keyword directly."""
    try:
        parameter = inspect.signature(callable_obj).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _select_embodiment(nodes: list[ArenaEnvGraphNodeSpec], assets_by_id: dict[str, Any]) -> Any | None:
    """Return the graph's embodiment entity, or None for scene-only environments."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphNodeType

    embodiment_nodes = [node for node in nodes if node.type == ArenaEnvGraphNodeType.EMBODIMENT]
    assert len(embodiment_nodes) <= 1, "Only one embodiment node can be converted to an IsaacLabArenaEnvironment"
    return assets_by_id[embodiment_nodes[0].id] if embodiment_nodes else None


def _apply_state_spatial_constraints(state_spec: ArenaEnvGraphStateSpec, assets_by_id: dict[str, Any]) -> None:
    """Attach relation entities or fixed poses to the environment's initial state."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType
    from isaaclab_arena.utils.pose import Pose

    relation_classes = spatial_constraint_relation_classes()
    for constraint in state_spec.spatial_constraints:
        params = dict(constraint.params)
        parent_asset = assets_by_id[constraint.parent]
        # Note(xinjieyao, 2026-05-26): at_pose constraint is a special case that is handled differently.
        # Ideally it shall be handled within the placer module.
        if constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSE:
            position_xyz = params.pop("position_xyz", None)
            rotation_xyzw = params.pop("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))
            assert position_xyz is not None, f"at_pose constraint '{constraint.id}' requires params.position_xyz"
            assert not params, f"Unsupported at_pose params for constraint '{constraint.id}': {sorted(params)}"
            _set_initial_pose(
                parent_asset,
                Pose(position_xyz=tuple(position_xyz), rotation_xyzw=tuple(rotation_xyzw)),
                constraint,
            )
            continue

        relation_cls = relation_classes.get(constraint.type)
        if relation_cls is None:
            raise NotImplementedError(f"Unsupported spatial constraint type '{constraint.type.value}'")
        params = _relation_params_for_constraint(constraint)
        # unary relation: parent_asset is the only endpoint
        # non-unary relation: parent_asset and child_asset are both endpoints
        if relation_cls.is_unary():
            _add_relation(parent_asset, relation_cls(**params), constraint)
        else:
            _add_relation(_child_asset(constraint, assets_by_id), relation_cls(parent_asset, **params), constraint)


def _relation_params_for_constraint(constraint: ArenaEnvGraphSpatialConstraintSpec) -> dict[str, Any]:
    """Translate YAML-friendly constraint params into relation constructor kwargs."""
    from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpatialConstraintType

    params = dict(constraint.params)
    if constraint.type == ArenaEnvGraphSpatialConstraintType.NEXT_TO and isinstance(params.get("side"), str):
        from isaaclab_arena.relations.relations import Side

        params["side"] = Side(params["side"])
    if constraint.type == ArenaEnvGraphSpatialConstraintType.AT_POSITION and "position_xyz" in params:
        position_xyz = params.pop("position_xyz")
        params.setdefault("x", position_xyz[0])
        params.setdefault("y", position_xyz[1])
        params.setdefault("z", position_xyz[2])
    return params


def _child_asset(constraint: ArenaEnvGraphSpatialConstraintSpec, assets_by_id: dict[str, Any]) -> Any:
    """Look up the child asset for a binary relation and name the bad constraint on failure."""
    assert constraint.child is not None, f"Constraint '{constraint.id}' requires a child node"
    return assets_by_id[constraint.child]


def _add_relation(asset: Any, relation: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Attach a relation to the asset that owns it."""
    assert hasattr(asset, "add_relation"), f"Constraint '{constraint.id}' target cannot accept relations"
    asset.add_relation(relation)


def _set_initial_pose(asset: Any, pose: Any, constraint: ArenaEnvGraphSpatialConstraintSpec) -> None:
    """Apply an at_pose constraint using the asset's initial-pose hook."""
    assert constraint.child is None, f"at_pose constraint '{constraint.id}' must not define a child node"
    assert hasattr(asset, "set_initial_pose"), f"Constraint '{constraint.id}' target cannot set an initial pose"
    asset.set_initial_pose(pose)
