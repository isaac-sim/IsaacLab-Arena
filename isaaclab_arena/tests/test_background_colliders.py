# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    """Write a background USD with an anchor, near/far fixtures, a wall, and a room-scale shell."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _make_box(stage, "table", center=(0.0, 0.0, 0.45), extents=(2.0, 1.0, 1.0))  # anchor; top intrudes region
    _make_box(stage, "near", center=(0.0, 0.0, 1.1), extents=(0.2, 0.2, 0.4))  # inside region
    _make_box(stage, "far", center=(5.0, 5.0, 0.5), extents=(0.3, 0.3, 1.0))  # outside region -> culled
    _make_box(stage, "wall_back", center=(0.0, 1.5, 1.0), extents=(2.0, 0.1, 2.0))  # behind region -> culled
    _make_box(stage, "room", center=(0.0, 0.0, 1.0), extents=(20.0, 20.0, 6.0))  # encloses region -> dropped
    stage.GetRootLayer().Save()


def _test_build_placement_region(simulation_app) -> bool:
    """Region pads the anchor footprint by the object's XY reach and rises by its height + clearance."""
    from unittest.mock import MagicMock

    from isaaclab_arena.relations.background_colliders import build_placement_region
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    anchor = MagicMock()
    anchor.get_world_bounding_box.return_value = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 1.0, 0.9))
    obj = MagicMock()
    obj.get_bounding_box.return_value = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.1, 0.1, 0.3))

    region = build_placement_region([anchor], [obj], clearance_m=0.05)
    # xy_pad = |(0.1, 0.1)| / 2 + 0.05 = 0.0707 + 0.05; top rises by object height (0.3) + clearance.
    xy_pad = 0.12071
    return (
        abs(region.min_point[0, 0].item() - -xy_pad) < 1e-4
        and abs(region.min_point[0, 2].item()) < 1e-6
        and abs(region.max_point[0, 0].item() - (2.0 + xy_pad)) < 1e-4
        and abs(region.max_point[0, 2].item() - (0.9 + 0.3 + 0.05)) < 1e-4
    )


def _test_find_background_colliders(simulation_app) -> bool:
    """Discovery keeps in-region fixtures; drops out-of-region ones, the enclosing shell, and the anchor."""
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


def _test_find_background_colliders_recurses_to_components(simulation_app) -> bool:
    """Auto-discovery descends group models to their component children, not the group as one box."""
    import tempfile
    from pathlib import Path

    from pxr import Kind, Usd, UsdGeom

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_colliders import find_background_colliders
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    usd_path = str(Path(tempfile.mkdtemp()) / "kinded_kitchen.usd")
    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    # A valid model hierarchy requires every ancestor of a component to be a group.
    Usd.ModelAPI(world.GetPrim()).SetKind(Kind.Tokens.assembly)
    cabinets = UsdGeom.Xform.Define(stage, "/World/cabinets")
    Usd.ModelAPI(cabinets.GetPrim()).SetKind(Kind.Tokens.group)
    _make_box(stage, "cabinets/sink", center=(0.0, 0.0, 1.1), extents=(0.2, 0.2, 0.4))  # component, in region
    Usd.ModelAPI(stage.GetPrimAtPath("/World/cabinets/sink")).SetKind(Kind.Tokens.component)
    _make_box(stage, "loose", center=(0.3, 0.0, 1.1), extents=(0.2, 0.2, 0.4))  # un-kinded top-level, in region
    stage.GetRootLayer().Save()

    background = Background(name="kinded_kitchen", usd_path=usd_path, object_min_z=-0.2, initial_pose=Pose.identity())
    region = AxisAlignedBoundingBox((-1.0, -0.5, 0.9), (1.0, 0.5, 1.4))
    colliders = find_background_colliders(background, region)
    return sorted(c.name for c in colliders) == ["loose", "sink"]


def _test_find_background_colliders_explicit_prim_paths(simulation_app) -> bool:
    """Explicit prim paths bypass auto-discovery but are still culled to the region."""
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


def _test_find_background_colliders_nested_prim_path(simulation_app) -> bool:
    """Explicit prim paths resolve nested groups, not just direct children of the default prim."""
    import tempfile
    from pathlib import Path

    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.background_colliders import find_background_colliders
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    usd_path = str(Path(tempfile.mkdtemp()) / "nested_kitchen.usd")
    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/group")
    _make_box(stage, "group/nested", center=(0.0, 0.0, 1.1), extents=(0.2, 0.2, 0.4))  # inside region
    stage.GetRootLayer().Save()

    background = Background(name="nested_kitchen", usd_path=usd_path, object_min_z=-0.2, initial_pose=Pose.identity())
    region = AxisAlignedBoundingBox((-1.0, -0.5, 0.9), (1.0, 0.5, 1.4))
    colliders = find_background_colliders(
        background, region, object_prim_paths=["{ENV_REGEX_NS}/nested_kitchen/group/nested"]
    )
    return [c.name for c in colliders] == ["nested"]


def _write_single_box_background(usd_path, center, extents):
    """Write a background USD whose default prim holds one box named 'widget'."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(usd_path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _make_box(stage, "widget", center=center, extents=extents)
    stage.GetRootLayer().Save()


def _test_object_reference_world_bbox_no_parent_pose(simulation_app) -> bool:
    """No parent pose (the default for backgrounds): world box is the local box at its own position."""
    import tempfile
    from pathlib import Path

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference

    usd_path = str(Path(tempfile.mkdtemp()) / "bg.usd")
    _write_single_box_background(usd_path, center=(0.0, 0.0, 1.1), extents=(0.2, 0.2, 0.4))
    background = Background(name="bg", usd_path=usd_path, object_min_z=-0.2)
    ref = ObjectReference(name="widget", prim_path="{ENV_REGEX_NS}/bg/widget", parent_asset=background)

    wbb = ref.get_world_bounding_box()
    return _close(wbb.min_point[0].tolist(), [-0.1, -0.1, 0.9]) and _close(wbb.max_point[0].tolist(), [0.1, 0.1, 1.3])


def _test_object_reference_world_bbox_yaw_parent(simulation_app) -> bool:
    """A 90° Z parent rotation swaps the box's X/Y extents (asymmetric box) and leaves Z unchanged."""
    import tempfile
    from pathlib import Path

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.utils.pose import Pose

    usd_path = str(Path(tempfile.mkdtemp()) / "bg.usd")
    _write_single_box_background(usd_path, center=(0.5, 0.0, 1.1), extents=(0.4, 0.2, 0.6))
    yaw_90 = Pose(position_xyz=(1.0, 2.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.7071067811865476, 0.7071067811865476))
    background = Background(name="bg", usd_path=usd_path, object_min_z=-0.2, initial_pose=yaw_90)
    ref = ObjectReference(name="widget", prim_path="{ENV_REGEX_NS}/bg/widget", parent_asset=background)

    wbb = ref.get_world_bounding_box()
    return (
        _close(wbb.size[0].tolist(), [0.2, 0.4, 0.6])  # X/Y swapped by the yaw
        and abs(wbb.min_point[0, 2].item() - 0.8) < 1e-4
        and abs(wbb.max_point[0, 2].item() - 1.4) < 1e-4
    )


def _close(actual, expected, tol=1e-4):
    """True if every component of two length-3 sequences is within tol."""
    return all(abs(a - e) < tol for a, e in zip(actual, expected))


def test_build_placement_region():
    result = run_simulation_app_function(_test_build_placement_region, headless=HEADLESS)
    assert result, "build_placement_region produced the wrong padded region"


def test_object_reference_world_bbox_no_parent_pose():
    result = run_simulation_app_function(_test_object_reference_world_bbox_no_parent_pose, headless=HEADLESS)
    assert result, "get_world_bounding_box with no parent pose returned the wrong box"


def test_object_reference_world_bbox_yaw_parent():
    result = run_simulation_app_function(_test_object_reference_world_bbox_yaw_parent, headless=HEADLESS)
    assert result, "get_world_bounding_box did not apply the parent's 90° yaw"


def test_find_background_colliders():
    result = run_simulation_app_function(_test_find_background_colliders, headless=HEADLESS)
    assert result, "find_background_colliders returned the wrong fixtures"


def test_find_background_colliders_recurses_to_components():
    result = run_simulation_app_function(_test_find_background_colliders_recurses_to_components, headless=HEADLESS)
    assert result, "auto-discovery did not recurse group models to component granularity"


def test_find_background_colliders_explicit_prim_paths():
    result = run_simulation_app_function(_test_find_background_colliders_explicit_prim_paths, headless=HEADLESS)
    assert result, "explicit prim-path discovery returned the wrong fixtures"


def test_find_background_colliders_nested_prim_path():
    result = run_simulation_app_function(_test_find_background_colliders_nested_prim_path, headless=HEADLESS)
    assert result, "nested explicit prim-path discovery returned the wrong fixtures"
