# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pathlib

from pxr import Gf, Usd, UsdGeom

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage
from isaaclab_arena.utils.usd_helpers import open_stage


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


def rename_rigid_body(stage: Usd.Stage, new_name: str) -> None:
    shallowest_rigid_body = find_shallowest_rigid_body_from_stage(stage)
    prim = stage.GetPrimAtPath(shallowest_rigid_body)
    assert prim.IsValid()
    prim_spec = stage.GetRootLayer().GetPrimAtPath(shallowest_rigid_body)
    prim_spec.name = new_name


def set_default_prim_to_rigid_body(stage: Usd.Stage) -> None:
    """Set the stage default prim to the shallowest rigid body.

    When the cached USD is referenced by the spawner at the proto path
    (/World/Template/Asset_0000), the referenced prim gets the default prim's
    content. So the proto root will have RigidBodyAPI and
    activate_contact_sensors(proto_root) will succeed.
    """
    rigid_body_path = find_shallowest_rigid_body_from_stage(stage)
    print(f"rigid_body_path: {rigid_body_path}")
    assert rigid_body_path is not None, "No rigid body found in stage"
    prim = stage.GetPrimAtPath(rigid_body_path)
    assert prim.IsValid()
    stage.SetDefaultPrim(prim)


def rescale_rename_rigid_body_and_save_to_cache(asset: Asset) -> str:
    cache_path = get_object_set_asset_cache_path(asset, asset.scale)
    with open_stage(asset.usd_path) as stage:
        rescale_root(stage, asset)
        rename_rigid_body(stage, new_name="rigid_body")
        set_default_prim_to_rigid_body(stage)
        stage.Export(str(cache_path))
    return str(cache_path)
