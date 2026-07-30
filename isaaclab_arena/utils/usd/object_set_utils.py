# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pathlib

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage
from isaaclab_arena.utils.usd_helpers import open_stage

CONTAINER_PRIM_NAME = "object_set_member"
"""Name of the Xform inserted above a root-level rigid body to give it a container to hang under."""


def get_object_set_asset_cache_path(asset: Asset, scale: tuple[float, float, float] | None = None) -> pathlib.Path:
    cache_dir = get_arena_asset_cache_dir()
    if scale is not None:
        scale_str = "_".join([str(s) for s in scale])
        return cache_dir / f"{asset.name}_{scale_str}.usd"
    else:
        return cache_dir / f"{asset.name}.usd"


def rescale_root(stage: Usd.Stage, asset: Asset) -> None:
    root_prim = stage.GetDefaultPrim()
    xformable = UsdGeom.Xformable(root_prim)
    scale_attr = root_prim.GetAttribute("xformOp:scale")
    if scale_attr.IsValid():
        UsdGeom.XformOp(scale_attr).Set(Gf.Vec3f(*asset.scale))
    else:
        xformable.AddScaleOp().Set(Gf.Vec3f(*asset.scale))


def rename_rigid_body(stage: Usd.Stage, new_name: str) -> str:
    """Rename the shallowest rigid body prim to new_name. Returns the path before rename (old path)."""
    shallowest_rigid_body = find_shallowest_rigid_body_from_stage(stage)
    prim = stage.GetPrimAtPath(shallowest_rigid_body)
    assert prim.IsValid()
    prim_spec = stage.GetRootLayer().GetPrimAtPath(shallowest_rigid_body)
    prim_spec.name = new_name
    return shallowest_rigid_body


def _rewrite_path_targets_in_root_layer(
    stage: Usd.Stage,
    old_rigid_body_path: str,
    new_rigid_body_path: str,
) -> None:
    """Rewrite relationship targets and attribute connections so they stay inside the reference.

    After renaming the root prim to "rigid_body", targets like </ranch_dressing/Looks/Material>
    would still point at the old path. When this file is referenced, USD only exposes prims under
    the default prim (rigid_body), so those targets are "outside the scope" and get ignored (grey
    materials, missing bindings). We author overrides on the root layer so all such paths use
    the new path (e.g. </rigid_body/Looks/Material>), keeping materials and connections valid.
    """
    new_path = Sdf.Path(new_rigid_body_path)
    # Prefixes for "path under rigid body" checks and for rewriting (e.g. /old/ -> /new/)
    old_prefix = old_rigid_body_path if old_rigid_body_path.endswith("/") else old_rigid_body_path + "/"
    new_prefix = new_rigid_body_path if new_rigid_body_path.endswith("/") else new_rigid_body_path + "/"
    layer = stage.GetRootLayer()
    edit_target = stage.GetEditTarget()
    stage.SetEditTarget(layer)  # Author all rewrites on the root layer so they export

    def replace_path(path: Sdf.Path) -> Sdf.Path | None:
        # If path equals or is under old_rigid_body_path, return the same path under new_rigid_body_path; else None.
        s = path.pathString
        if s == old_rigid_body_path:
            return new_path  # Exact match: rigid body root
        if s.startswith(old_prefix):
            # Path under old root: keep relative tail, reparent under new root
            return Sdf.Path(new_prefix.rstrip("/") + s[len(old_rigid_body_path) :])
        return None

    try:
        for prim in stage.Traverse():
            # Rewrite relationship targets (e.g. material:binding -> </old_path/Looks/...>)
            for rel in prim.GetRelationships():
                targets = list(rel.GetTargets())
                if not targets:
                    continue
                new_targets = []
                changed = False
                for t in targets:
                    r = replace_path(t)
                    if r is not None:
                        new_targets.append(r)
                        changed = True
                    else:
                        new_targets.append(t)
                if changed:
                    rel.SetTargets(new_targets)
            # Rewrite attribute connections (e.g. shader inputs that point at prim paths)
            for attr in prim.GetAttributes():
                if not getattr(attr, "HasAuthoredConnections", lambda: False)():
                    continue
                try:
                    conns = attr.GetConnections()
                except Exception:
                    continue
                if not conns:
                    continue
                new_conns = []
                changed = False
                for c in conns:
                    r = replace_path(c) if isinstance(c, Sdf.Path) else None
                    if r is not None:
                        new_conns.append(r)
                        changed = True
                    else:
                        new_conns.append(c)
                if changed and getattr(attr, "SetConnections", None) is not None:
                    attr.SetConnections(new_conns)
    finally:
        stage.SetEditTarget(edit_target)


def _wrap_root_rigid_body_in_container(stage: Usd.Stage) -> None:
    """Reparent a root-level rigid body under a container Xform, leaving deeper ones alone."""
    rigid_body_path = find_shallowest_rigid_body_from_stage(stage)
    assert rigid_body_path is not None, "No rigid body found in stage"
    if rigid_body_path.count("/") > 1:
        return

    old_path = Sdf.Path(rigid_body_path)
    container_path = Sdf.Path(f"/{CONTAINER_PRIM_NAME}")
    assert not stage.GetPrimAtPath(container_path), f"Stage already has a prim at {container_path}"
    new_path = container_path.AppendChild(old_path.name)

    layer = stage.GetRootLayer()
    container_spec = Sdf.CreatePrimInLayer(layer, container_path)
    container_spec.specifier = Sdf.SpecifierDef
    container_spec.typeName = "Xform"
    assert Sdf.CopySpec(layer, old_path, layer, new_path), f"Failed to reparent {old_path} under {container_path}"
    del layer.rootPrims[old_path.name]

    stage.SetDefaultPrim(stage.GetPrimAtPath(container_path))
    # The copied specs still carry targets pointing at the pre-move path; keep them in scope.
    _rewrite_path_targets_in_root_layer(stage, str(old_path), str(new_path))


def _set_default_prim_for_object_set_cache(stage: Usd.Stage) -> None:
    """Set the stage default prim to the rigid body's parent.

    Referencing this file exposes the default prim, so putting the parent there leaves every member
    with its rigid body at the same path, "/rigid_body". It also gives activate_contact_sensors what
    it looks for: a prim with the rigid body under it, rather than the rigid body itself.
    """
    rigid_body_path = find_shallowest_rigid_body_from_stage(stage)
    assert rigid_body_path is not None, "No rigid body found in stage"
    assert rigid_body_path.count("/") > 1, f"Rigid body {rigid_body_path!r} must be wrapped in a container first"
    default_prim_path = str(Sdf.Path(rigid_body_path).GetParentPath())
    prim = stage.GetPrimAtPath(default_prim_path)
    assert prim.IsValid(), f"Default prim path {default_prim_path!r} is not valid"
    stage.SetDefaultPrim(prim)


def rescale_rename_rigid_body_and_save_to_cache(asset: Asset) -> str:
    """Export a scaled, compatible USD to the asset cache for use in object sets.

    Object sets need all member USDs to share the same structure (same rigid-body path) and
    scale. This function: (1) rescales the root, (2) moves a root-level rigid body under a
    container so every member nests it the same way, (3) renames the shallowest rigid body to
    "rigid_body", (4) rewrites relationship/connection targets to use that path so they remain
    valid when the file is referenced, (5) sets the default prim to the rigid body's parent,
    (6) exports to the cache. Without step (4), material bindings and shader connections would
    point at the old root path and be ignored by USD, causing grey/missing materials.
    """
    cache_path = get_object_set_asset_cache_path(asset, asset.scale)
    with open_stage(asset.usd_path) as stage:
        rescale_root(stage, asset)
        # Move root-level rigid body under a container so every member nests it the same way.
        _wrap_root_rigid_body_in_container(stage)
        # Unify name; need old path for rewrite
        old_rb_path = rename_rigid_body(stage, new_name="rigid_body")
        # Path after rename (e.g. /rigid_body or /root/rigid_body)
        new_rb_path = find_shallowest_rigid_body_from_stage(stage)
        # Keep materials/connections in scope
        _rewrite_path_targets_in_root_layer(stage, old_rb_path, new_rb_path)
        # So contact sensors are found when cache is referenced
        _set_default_prim_for_object_set_cache(stage)
        stage.Export(str(cache_path))
    return str(cache_path)
