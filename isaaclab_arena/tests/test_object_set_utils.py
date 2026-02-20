# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_rescale_rename_rigid_body_and_save_to_cache_depth0(simulation_app):
    """Test cache pipeline with a depth-0 rigid body (single root prim)."""
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.utils.usd.object_set_utils import (
        get_object_set_asset_cache_path,
        rescale_rename_rigid_body_and_save_to_cache,
    )
    from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body_from_stage

    # Create minimal USD: single root prim with RigidBodyAPI (depth 0)
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/original_rb", "Xform")
    prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
    stage.SetDefaultPrim(prim)
    with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as f:
        src_path = f.name
    stage.Export(src_path)
    stage = None

    class _MinimalAsset:
        name = "test_depth0_asset"
        usd_path = src_path
        scale = (1.0, 1.0, 1.0)

    try:
        cache_path_str = rescale_rename_rigid_body_and_save_to_cache(_MinimalAsset())
        assert os.path.isfile(cache_path_str), f"Cache file not created: {cache_path_str}"

        cache_stage = Usd.Stage.Open(cache_path_str)
        assert cache_stage is not None
        default_prim = cache_stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Default prim not set"
        # Depth 0: default prim is the rigid body itself
        assert default_prim.GetPath().pathString == "/rigid_body"
        rb_path = find_shallowest_rigid_body_from_stage(cache_stage)
        assert rb_path == "/rigid_body"
        rb_prim = cache_stage.GetPrimAtPath("/rigid_body")
        assert rb_prim.IsValid() and rb_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        return True
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        cache_path = get_object_set_asset_cache_path(_MinimalAsset(), (1.0, 1.0, 1.0))
        try:
            os.unlink(cache_path)
        except OSError:
            pass


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
        try:
            os.unlink(src_path)
        except OSError:
            pass
        cache_path = get_object_set_asset_cache_path(_MinimalAsset(), (2.0, 2.0, 2.0))
        try:
            os.unlink(cache_path)
        except OSError:
            pass


def test_rescale_rename_rigid_body_and_save_to_cache_depth0():
    result = run_simulation_app_function(
        _test_rescale_rename_rigid_body_and_save_to_cache_depth0,
        headless=HEADLESS,
    )
    assert result, "test_rescale_rename_rigid_body_and_save_to_cache_depth0 failed"


def test_rescale_rename_rigid_body_and_save_to_cache_depth1():
    result = run_simulation_app_function(
        _test_rescale_rename_rigid_body_and_save_to_cache_depth1,
        headless=HEADLESS,
    )
    assert result, "test_rescale_rename_rigid_body_and_save_to_cache_depth1 failed"


if __name__ == "__main__":
    test_rescale_rename_rigid_body_and_save_to_cache_depth0()
    test_rescale_rename_rigid_body_and_save_to_cache_depth1()
