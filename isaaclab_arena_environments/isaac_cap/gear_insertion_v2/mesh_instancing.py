# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in OpenUSD mesh instancing for industrial benchmark assets."""

from __future__ import annotations

import hashlib

import isaaclab.sim as sim_utils
from isaaclab.sim import MultiUsdFileCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.spawners.wrappers.wrappers import spawn_multi_usd_file
from isaaclab.sim.utils import clone
from pxr import Sdf, Usd, UsdGeom, UsdShade

_STANDARD_USD_SPAWNERS = {
    "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd",
    from_files.spawn_from_usd,
}
_STANDARD_MULTI_USD_SPAWNERS = {
    "isaaclab.sim.spawners.wrappers.wrappers:spawn_multi_usd_file",
    spawn_multi_usd_file,
}


def make_referenced_mesh_leaves_instanceable(root_prim: Usd.Prim) -> None:
    """Instance robot render geometry while keeping its physics tree writable.

    Directly marking a referenced ``Mesh`` instanceable makes articulated robot
    geometry disappear in Isaac RTX's Fabric render path.  Keep the original
    mesh as the physics-owned prim, hide only its render purpose, and add an
    instanceable sibling ``Xform`` that references a shared geometry prototype.
    This is the same scene-graph boundary used by authored instanceable assets:
    per-link transforms stay unique while immutable mesh topology is shared.
    """
    meshes = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAuthoredReferences()
        and UsdGeom.Imageable(prim).ComputePurpose() in ("default", "render")
    ]
    for mesh in meshes:
        instance_path = mesh.GetPath().GetParentPath().AppendChild(f"{mesh.GetName()}__render_instance")
        if mesh.GetStage().GetPrimAtPath(instance_path).IsValid():
            continue

        reference, reference_layer = _first_authored_reference(mesh)
        if reference is None or reference_layer is None:
            continue
        asset_path = reference.assetPath
        if asset_path:
            asset_path = Sdf.ComputeAssetPathRelativeToLayer(reference_layer, asset_path)

        stage = mesh.GetStage()
        prototype_key = f"{asset_path}|{reference.primPath}"
        prototype_name = hashlib.sha256(prototype_key.encode()).hexdigest()[:16]
        prototype_path = Sdf.Path(f"/World/__IndustrialRobotGeometry/p_{prototype_name}")
        prototype = stage.GetPrimAtPath(prototype_path)
        if not prototype.IsValid():
            stage.CreateClassPrim(prototype_path)
            prototype_mesh = stage.DefinePrim(prototype_path.AppendChild("mesh"), "Mesh")
            if asset_path:
                prototype_mesh.GetReferences().AddReference(asset_path, reference.primPath)
            else:
                prototype_mesh.GetReferences().AddInternalReference(reference.primPath)

        render_instance = stage.DefinePrim(instance_path, "Xform")
        render_instance.GetReferences().AddInternalReference(prototype_path)
        render_instance.SetInstanceable(True)
        local_transform = UsdGeom.Xformable(mesh).GetLocalTransformation()
        UsdGeom.Xformable(render_instance).AddTransformOp().Set(local_transform)

        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if material:
            UsdShade.MaterialBindingAPI.Apply(render_instance).Bind(material)
        UsdGeom.Imageable(mesh).MakeInvisible()


def _first_authored_reference(
    prim: Usd.Prim,
) -> tuple[Sdf.Reference | None, Sdf.Layer | None]:
    """Return the strongest authored reference and the layer that owns it."""
    for spec in prim.GetPrimStack():
        references = (
            list(spec.referenceList.prependedItems)
            + list(spec.referenceList.explicitItems)
            + list(spec.referenceList.addedItems)
            + list(spec.referenceList.appendedItems)
        )
        if references:
            return references[0], spec.layer
    return None, None


def make_asset_root_instanceable(root_prim: Usd.Prim) -> None:
    """Instance a referenced asset root, falling back to referenced mesh leaves."""
    if root_prim.HasAuthoredReferences() or root_prim.HasAuthoredPayloads():
        root_prim.SetInstanceable(True)
    else:
        make_referenced_mesh_leaves_instanceable(root_prim)


@clone
def spawn_usd_with_instanceable_meshes(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD and instance its referenced mesh leaves before env cloning."""
    root_prim = from_files._spawn_from_usd_file(
        prim_path,
        cfg.usd_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    make_referenced_mesh_leaves_instanceable(root_prim)
    return root_prim


def spawn_usd_as_instance(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD normally, then make every resolved asset root an instance."""
    root_prim = from_files.spawn_from_usd(
        prim_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    stage = root_prim.GetStage()
    for path in sim_utils.find_matching_prim_paths(prim_path):
        make_asset_root_instanceable(stage.GetPrimAtPath(path))
    return root_prim


def _multi_usd_spawn_paths(prim_path: str, cfg: MultiUsdFileCfg) -> list[str]:
    if cfg.spawn_paths is not None:
        return [path for path in cfg.spawn_paths if path is not None]
    split_path = sim_utils.split_path_expr(prim_path)
    prefix_path, base_name = "/".join(split_path[:-1]), split_path[-1]
    base_glob = sim_utils.path_expr_to_glob(base_name)
    usd_paths = [cfg.usd_path] if isinstance(cfg.usd_path, str) else cfg.usd_path
    return [f"{prefix_path}/{base_glob.replace('*', str(index))}" for index in range(len(usd_paths))]


def spawn_multi_usd_as_instances(
    prim_path: str,
    cfg: MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn each selected heterogeneous USD as an independent instance."""
    root_prim = spawn_multi_usd_file(
        prim_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    stage = root_prim.GetStage()
    for path_expression in _multi_usd_spawn_paths(prim_path, cfg):
        matching_paths = sim_utils.find_matching_prim_paths(path_expression, stage=stage)
        if not matching_paths:
            raise RuntimeError(f"spawned multi-USD asset path did not resolve: {path_expression}")
        for path in matching_paths:
            asset_prim = stage.GetPrimAtPath(path)
            make_asset_root_instanceable(asset_prim)
    return root_prim


def configure_usd_asset_instancing(spawn_cfg: object, enabled: bool) -> None:
    """Select root instancing for an otherwise standard USD asset spawner."""
    if not isinstance(enabled, bool):
        raise TypeError("use_instanceable_meshes must be a boolean")
    if not isinstance(spawn_cfg, UsdFileCfg):
        return
    if spawn_cfg.deformable_props is not None or spawn_cfg.make_uninstanceable:
        return
    if isinstance(spawn_cfg, MultiUsdFileCfg):
        if spawn_cfg.func in _STANDARD_MULTI_USD_SPAWNERS or spawn_cfg.func is spawn_multi_usd_as_instances:
            spawn_cfg.func = spawn_multi_usd_as_instances if enabled else spawn_multi_usd_file
    elif spawn_cfg.func in _STANDARD_USD_SPAWNERS or spawn_cfg.func is spawn_usd_as_instance:
        spawn_cfg.func = spawn_usd_as_instance if enabled else from_files.spawn_from_usd


def configure_scene_mesh_instancing(scene, enabled: bool) -> None:
    """Instance standalone USD assets while preserving referenced parent trees."""
    if not isinstance(enabled, bool):
        raise TypeError("use_instanceable_meshes must be a boolean")
    referenced_parents = {
        id(parent) for asset in scene.assets.values() if (parent := getattr(asset, "parent_asset", None)) is not None
    }
    for asset in scene.assets.values():
        if id(asset) in referenced_parents:
            continue
        object_cfg = getattr(asset, "object_cfg", None)
        configure_usd_asset_instancing(getattr(object_cfg, "spawn", None), enabled)


__all__ = [
    "configure_scene_mesh_instancing",
    "configure_usd_asset_instancing",
    "make_asset_root_instanceable",
    "make_referenced_mesh_leaves_instanceable",
    "spawn_multi_usd_as_instances",
    "spawn_usd_as_instance",
    "spawn_usd_with_instanceable_meshes",
]
