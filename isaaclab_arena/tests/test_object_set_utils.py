# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import tempfile

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_rescale_rename_rigid_body_and_save_to_cache_depth0(simulation_app):
    """Test cache pipeline with a depth-0 rigid body (single root prim)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.object_set_utils import (
        CONTAINER_PRIM_NAME,
        get_object_set_asset_cache_path,
        rescale_rename_rigid_body_and_save_to_cache,
    )
    from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage

    # Create minimal USD: single root prim with RigidBodyAPI (depth 0), already carrying a scale
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/original_rb", "Xform")
    prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 3.0))
    stage.SetDefaultPrim(prim)
    with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
        src_path = f.name
    stage.Export(src_path)
    stage = None

    class _MinimalAsset:
        name = "test_depth0_asset"
        usd_path = src_path
        scale = (2.0, 2.0, 2.0)

    try:
        cache_path_str = rescale_rename_rigid_body_and_save_to_cache(_MinimalAsset())
        assert os.path.isfile(cache_path_str), f"Cache file not created: {cache_path_str}"

        cache_stage = Usd.Stage.Open(cache_path_str)
        assert cache_stage is not None
        default_prim = cache_stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Default prim not set"
        # Depth 0: the rigid body is wrapped so it hangs under the default prim like deeper members
        assert default_prim.GetPath().pathString == f"/{CONTAINER_PRIM_NAME}"
        rb_path = find_shallowest_rigid_body_from_stage(cache_stage)
        assert rb_path == f"/{CONTAINER_PRIM_NAME}/rigid_body"
        rb_prim = cache_stage.GetPrimAtPath(rb_path)
        assert rb_prim.IsValid() and rb_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        # The asset scale replaces the one already on the root; the container must not add another,
        # which would apply both and leave the member 3x too small.
        assert rb_prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3f(2.0, 2.0, 2.0)
        assert not default_prim.GetAttribute("xformOp:scale").IsValid(), "Container must stay identity"
        return True
    finally:
        with contextlib.suppress(OSError):
            os.unlink(src_path)
        cache_path = get_object_set_asset_cache_path(_MinimalAsset(), (2.0, 2.0, 2.0))
        with contextlib.suppress(OSError):
            os.unlink(cache_path)


def _test_rescale_rename_rigid_body_and_save_to_cache_depth1(simulation_app):
    """Test cache pipeline with a depth-1 rigid body (rigid body under a root scope)."""
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.utils.usd.object_set_utils import (
        get_object_set_asset_cache_path,
        rescale_rename_rigid_body_and_save_to_cache,
    )
    from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage

    # Create minimal USD: /root/original_rb with RigidBodyAPI (depth 1)
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Scope")
    rb_prim = stage.DefinePrim("/root/original_rb", "Xform")
    rb_prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
    stage.SetDefaultPrim(root)
    with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
        src_path = f.name
    stage.Export(src_path)
    stage = None

    class _MinimalAsset:
        name = "test_depth1_asset"
        usd_path = src_path
        scale = (2.0, 2.0, 2.0)

    try:
        cache_path_str = rescale_rename_rigid_body_and_save_to_cache(_MinimalAsset())
        assert os.path.isfile(cache_path_str), f"Cache file not created: {cache_path_str}"

        cache_stage = Usd.Stage.Open(cache_path_str)
        assert cache_stage is not None
        default_prim = cache_stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Default prim not set"
        # Depth 1: default prim is the parent so referenced prim is a scope with rigid_body as child
        assert default_prim.GetPath().pathString == "/root"
        rb_path = find_shallowest_rigid_body_from_stage(cache_stage)
        assert rb_path == "/root/rigid_body"
        rb_prim = cache_stage.GetPrimAtPath("/root/rigid_body")
        assert rb_prim.IsValid() and rb_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        # Scale should have been applied to root
        root_prim = cache_stage.GetPrimAtPath("/root")
        scale_attr = root_prim.GetAttribute("xformOp:scale")
        assert scale_attr.IsValid()
        from pxr import Gf

        assert scale_attr.Get() == Gf.Vec3f(2.0, 2.0, 2.0)
        return True
    finally:
        with contextlib.suppress(OSError):
            os.unlink(src_path)
        cache_path = get_object_set_asset_cache_path(_MinimalAsset(), (2.0, 2.0, 2.0))
        with contextlib.suppress(OSError):
            os.unlink(cache_path)


def _test_cache_pipeline_unifies_mixed_rigid_body_depths(simulation_app):
    """Members nesting their rigid bodies at different depths must still share one referenced path."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    from isaaclab_arena.utils.usd.object_set_utils import (
        get_object_set_asset_cache_path,
        rescale_rename_rigid_body_and_save_to_cache,
    )

    class _MinimalAsset:
        def __init__(self, name: str, usd_path: str):
            self.name = name
            self.usd_path = usd_path
            self.scale = (1.0, 1.0, 1.0)

    def _export_asset(rigid_body_path: str) -> str:
        """Write a USD whose only rigid body sits at rigid_body_path, with a material bound to it."""
        stage = Usd.Stage.CreateInMemory()
        prefixes = Sdf.Path(rigid_body_path).GetPrefixes()
        for prefix in prefixes:
            stage.DefinePrim(prefix, "Xform")
        rb_prim = stage.GetPrimAtPath(rigid_body_path)
        rb_prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
        material = UsdShade.Material.Define(stage, f"{rigid_body_path}/Looks/Mat")
        UsdShade.MaterialBindingAPI.Apply(rb_prim).Bind(material)
        stage.SetDefaultPrim(stage.GetPrimAtPath(prefixes[0]))
        with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
            src_path = f.name
        stage.Export(src_path)
        return src_path

    rigid_body_paths = {"depth0": "/rb", "depth1": "/root/rb", "depth2": "/root/scope/rb"}
    assets = [_MinimalAsset(f"test_mixed_{name}", _export_asset(path)) for name, path in rigid_body_paths.items()]

    try:
        # Isaac Lab reads the rigid body path from the first environment and reuses it for the rest,
        # so referencing any member must put the rigid body at the same relative path.
        holder_path = "/World/member"
        relative_paths = set()
        for asset in assets:
            cache_path = rescale_rename_rigid_body_and_save_to_cache(asset)
            stage = Usd.Stage.CreateInMemory()
            holder = stage.DefinePrim(holder_path, "Xform")
            holder.GetReferences().AddReference(cache_path)
            rigid_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert len(rigid_bodies) == 1, f"{asset.name}: expected one rigid body, got {rigid_bodies}"
            relative_paths.add(rigid_bodies[0][len(holder_path) :])
            bound_material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(rigid_bodies[0])).ComputeBoundMaterial()
            assert bound_material and bound_material.GetPrim().IsValid(), f"{asset.name}: material binding lost"
        assert relative_paths == {"/rigid_body"}, f"Members disagree on the rigid body path: {relative_paths}"
        return True
    finally:
        for asset in assets:
            with contextlib.suppress(OSError):
                os.unlink(asset.usd_path)
            with contextlib.suppress(OSError):
                os.unlink(get_object_set_asset_cache_path(asset, asset.scale))


def test_rescale_rename_rigid_body_and_save_to_cache_depth0():
    result = run_function_with_persistent_simulation_app(
        _test_rescale_rename_rigid_body_and_save_to_cache_depth0,
        headless=HEADLESS,
    )
    assert result, "test_rescale_rename_rigid_body_and_save_to_cache_depth0 failed"


def test_rescale_rename_rigid_body_and_save_to_cache_depth1():
    result = run_function_with_persistent_simulation_app(
        _test_rescale_rename_rigid_body_and_save_to_cache_depth1,
        headless=HEADLESS,
    )
    assert result, "test_rescale_rename_rigid_body_and_save_to_cache_depth1 failed"


def test_cache_pipeline_unifies_mixed_rigid_body_depths():
    result = run_function_with_persistent_simulation_app(
        _test_cache_pipeline_unifies_mixed_rigid_body_depths,
        headless=HEADLESS,
    )
    assert result, "test_cache_pipeline_unifies_mixed_rigid_body_depths failed"


if __name__ == "__main__":
    test_rescale_rename_rigid_body_and_save_to_cache_depth0()
    test_rescale_rename_rigid_body_and_save_to_cache_depth1()
    test_cache_pipeline_unifies_mixed_rigid_body_depths()
