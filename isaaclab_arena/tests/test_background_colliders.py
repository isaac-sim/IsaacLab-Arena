# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for placement-region construction and background collider discovery."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _make_box(stage, name, center, extents):
    """Author a unit cube at /World/<name> scaled to extents and translated to center."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, f"/World/{name}")
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddScaleOp().Set(Gf.Vec3f(*extents))
    cube = UsdGeom.Cube.Define(stage, f"/World/{name}/geom")
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])


def _make_synthetic_background(usd_path):
    """Write a background USD with an anchor surface, a near/far fixture, and a wall."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _make_box(stage, "table", center=(0.0, 0.0, 0.45), extents=(2.0, 1.0, 1.0))  # top intrudes region
    _make_box(stage, "near", center=(0.0, 0.0, 1.1), extents=(0.2, 0.2, 0.4))  # inside region
    _make_box(stage, "far", center=(5.0, 5.0, 0.5), extents=(0.3, 0.3, 1.0))  # outside region
    _make_box(stage, "wall_back", center=(0.0, 1.5, 1.0), extents=(2.0, 0.1, 2.0))  # excluded by name
    stage.GetRootLayer().Save()


def _test_build_placement_region(simulation_app) -> bool:
    """Region spans the anchor footprint and rises by the tallest object's height + clearance."""
    from unittest.mock import MagicMock

    from isaaclab_arena.relations.background_colliders import build_placement_region
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    anchor = MagicMock()
    anchor.get_world_bounding_box.return_value = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 1.0, 0.9))
    obj = MagicMock()
    obj.get_bounding_box.return_value = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.1, 0.1, 0.3))

    region = build_placement_region([anchor], [obj], clearance_m=0.05)
    return (
        region.min_point.tolist() == [[0.0, 0.0, 0.0]]
        and region.max_point[0, 0].item() == 2.0
        and abs(region.max_point[0, 2].item() - (0.9 + 0.3 + 0.05)) < 1e-5
    )


def _test_find_background_colliders(simulation_app) -> bool:
    """Discovery keeps in-region fixtures, drops far ones, the wall (by name), and the anchor."""
    import tempfile
    from pathlib import Path

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.relations.background_colliders import find_background_colliders
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    usd_path = str(Path(tempfile.mkdtemp()) / "synthetic_kitchen.usd")
    _make_synthetic_background(usd_path)
    background = Background(
        name="synthetic_kitchen", usd_path=usd_path, object_min_z=-0.2, initial_pose=Pose.identity()
    )
    anchor = ObjectReference(name="table", prim_path="{ENV_REGEX_NS}/synthetic_kitchen/table", parent_asset=background)

    region = AxisAlignedBoundingBox((-1.0, -0.5, 0.9), (1.0, 0.5, 1.4))
    colliders = find_background_colliders(background, region, anchors=[anchor])
    return sorted(c.name for c in colliders) == ["near"]


def _test_find_background_colliders_explicit_prim_paths(simulation_app) -> bool:
    """Explicit prim paths skip name-based discovery but are still culled to the region."""
    import tempfile
    from pathlib import Path

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_colliders import find_background_colliders
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    usd_path = str(Path(tempfile.mkdtemp()) / "synthetic_kitchen.usd")
    _make_synthetic_background(usd_path)
    background = Background(
        name="synthetic_kitchen", usd_path=usd_path, object_min_z=-0.2, initial_pose=Pose.identity()
    )

    region = AxisAlignedBoundingBox((-1.0, -0.5, 0.9), (1.0, 0.5, 1.4))
    # "far" is supplied explicitly but sits outside the region, so region culling still drops it.
    prim_paths = [
        "{ENV_REGEX_NS}/synthetic_kitchen/near",
        "{ENV_REGEX_NS}/synthetic_kitchen/far",
    ]
    colliders = find_background_colliders(background, region, object_prim_paths=prim_paths)
    return sorted(c.name for c in colliders) == ["near"]


def test_build_placement_region():
    result = run_simulation_app_function(_test_build_placement_region, headless=HEADLESS)
    assert result, "build_placement_region produced the wrong region"


def test_find_background_colliders():
    result = run_simulation_app_function(_test_find_background_colliders, headless=HEADLESS)
    assert result, "find_background_colliders returned the wrong fixtures"


def test_find_background_colliders_explicit_prim_paths():
    result = run_simulation_app_function(_test_find_background_colliders_explicit_prim_paths, headless=HEADLESS)
    assert result, "explicit prim-path discovery returned the wrong fixtures"
