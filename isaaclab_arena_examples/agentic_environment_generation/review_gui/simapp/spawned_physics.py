# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect effective physics on an asset spawned through the Arena environment path."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
from isaaclab_arena.utils.usd.prim_tree import load_stage_prim_tree, usd_prim_records_to_dicts


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return _json_value(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    return repr(value)


def _configured_physics(asset) -> dict[str, Any]:
    spawned_asset = asset.get_spawned_asset()
    object_cfg = getattr(spawned_asset, "object_cfg", None)
    spawn = getattr(object_cfg, "spawn", None)
    if spawn is None:
        return {}
    names = (
        "scale",
        "rigid_props",
        "collision_props",
        "mass_props",
        "articulation_props",
        "prim_physics",
        "scale_articulation_prismatic_joints",
    )
    return {name: _json_value(getattr(spawn, name)) for name in names if hasattr(spawn, name)}


def _unapplied_override_findings(root_prim, asset, records) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    spawn = getattr(getattr(asset.get_spawned_asset(), "object_cfg", None), "spawn", None)
    if spawn is None:
        return findings

    for relative_path in getattr(spawn, "prim_physics", {}):
        target = root_prim.GetStage().GetPrimAtPath(root_prim.GetPath().AppendPath(relative_path))
        if not target or not target.IsValid():
            findings.append({
                "override": f"prim_physics[{relative_path!r}]",
                "reason": "spawned target prim is missing",
            })
        elif target.IsInstanceProxy():
            findings.append({
                "override": f"prim_physics[{relative_path!r}]",
                "reason": "spawned target remains an uneditable instance proxy",
            })

    colliders = [record.collider for record in records if record.collider is not None]
    collision_props = getattr(spawn, "collision_props", None)
    if collision_props is not None and not colliders:
        findings.append({"override": "collision_props", "reason": "spawned asset has no collision API"})

    mesh_props = getattr(collision_props, "mesh_collision_property", None)
    expected_approximation = getattr(mesh_props, "mesh_approximation_name", None)
    mesh_colliders = [collider for collider in colliders if collider.type_name == "Mesh"]
    if expected_approximation and mesh_colliders:
        mismatches = [
            collider.approximation for collider in mesh_colliders if collider.approximation != expected_approximation
        ]
        if mismatches:
            findings.append({
                "override": "collision_props.mesh_collision_property.mesh_approximation_name",
                "expected": expected_approximation,
                "effective_values": sorted({str(value) for value in mismatches}),
                "reason": "effective mesh approximation differs from the configured override",
            })

    rigid_props = getattr(spawn, "rigid_props", None)
    if rigid_props is not None and not any(record.rigid_body is not None for record in records):
        findings.append({"override": "rigid_props", "reason": "spawned asset has no rigid-body API"})
    return findings


def _horizontal_mesh_surfaces(root_prim) -> list[dict[str, Any]]:
    """Return upward mesh facets using a world-Z heuristic.

    This intentionally ignores primitive colliders and non-Z-up stages. It is an
    opt-in authoring hint, not a placement or collision correctness test.
    """
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim

    surfaces = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = extract_trimesh_from_prim(prim.GetStage(), str(prim.GetPath()))
        T_W_G = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        mesh.apply_transform(T_W_G)
        for faces, normal, area in zip(mesh.facets, mesh.facets_normal, mesh.facets_area):
            if float(normal[2]) < 0.999:
                continue
            vertices = mesh.triangles[faces].reshape(-1, 3)
            surfaces.append({
                "prim_path": str(prim.GetPath()),
                "height_world_z_m": float(vertices[:, 2].mean()),
                "area_m2": float(area),
                "bounds_xy_m": [
                    vertices[:, :2].min(axis=0).tolist(),
                    vertices[:, :2].max(axis=0).tolist(),
                ],
            })
    return surfaces


def run_spawned_physics_inspection(
    app,
    yaml_text: str,
    *,
    asset_id: str,
    prim_path: str | None = None,
    include_horizontal_surfaces: bool = False,
) -> dict[str, Any]:
    """Spawn one environment and report the selected asset's composed physics."""
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    teardown_simulation_app(suppress_exceptions=True, make_new_stage=True)
    graph_spec = ArenaEnvGraphSpec.model_validate(yaml.safe_load(yaml_text))
    arena_env = graph_spec.to_arena_env()
    assert asset_id in arena_env.scene.assets, f"Scene asset does not exist: {asset_id}"
    asset = arena_env.scene.assets[asset_id]
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=1,
            resolve_on_reset=False,
            disable_fabric=True,
            device="cpu",
        ),
    )
    env = None
    try:
        env_cfg, env_kwargs = builder.build_env_cfg()
        env = builder.make_registered(env_cfg, env_kwargs)
        env.reset()
        stage = env.unwrapped.scene.stage
        env_prim_path = env.unwrapped.scene.env_prim_paths[0]
        asset_prim_path = asset.get_prim_path().replace("{ENV_REGEX_NS}", env_prim_path)
        asset_root_prim = stage.GetPrimAtPath(asset_prim_path)
        assert asset_root_prim and asset_root_prim.IsValid(), f"Spawned asset prim does not exist: {asset_prim_path}"
        root_prim = asset_root_prim
        if prim_path:
            selected_path = prim_path if prim_path.startswith("/") else str(root_prim.GetPath().AppendPath(prim_path))
            root_prim = stage.GetPrimAtPath(selected_path)
            assert root_prim and root_prim.IsValid(), f"Spawned asset subtree does not exist: {prim_path}"

        records = load_stage_prim_tree(stage, root_prim, include_physical_root=True)
        report = {
            "asset_id": asset_id,
            "spawned_prim_path": str(root_prim.GetPath()),
            "configured_physics": _configured_physics(asset),
            "effective_physics": usd_prim_records_to_dicts(records),
            "unapplied_override_findings": _unapplied_override_findings(asset_root_prim, asset, records),
        }
        if include_horizontal_surfaces:
            report["horizontal_surfaces"] = _horizontal_mesh_surfaces(root_prim)
            report["horizontal_surface_assumptions"] = (
                "Mesh colliders only; world Z is treated as up; facets must have normal.z >= 0.999."
            )
        return report
    finally:
        if env is not None:
            with suppress(Exception):
                env.close()
        teardown_simulation_app(suppress_exceptions=True, make_new_stage=True)
        with suppress(Exception):
            app.update()
