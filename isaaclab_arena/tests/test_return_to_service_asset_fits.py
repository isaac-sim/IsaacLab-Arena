# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject assembly jams using exported collision geometry and placement metadata."""

import pytest


def _obstruction_cell(plug_depth=0.030, plug_x=0.001, tab_x=0.025, socket_y=0.0, cup_intruder=False):
    """Author collision primitives reproducing the supported plug and segmented inlet."""
    import math

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features

    cup = Usd.Stage.CreateInMemory()
    for index in range(16):
        angle = math.tau * index / 16
        center = (0.055, 0.0135 * math.cos(angle), 0.037 + 0.0135 * math.sin(angle))
        box = _box(cup, f"/Cup/Inlet{index}", center, (0.026, 0.003, 0.024 * math.tan(math.pi / 16)), "inlet")
        translation, scale = box.GetOrderedXformOps()
        rotation = box.AddRotateXOp()
        rotation.Set(math.degrees(angle))
        box.SetXformOpOrder((translation, rotation, scale))
    if cup_intruder:
        _box(cup, "/Cup/Intruder", (0.048, 0, 0.037), (0.002, 0.004, 0.004), "front_bulkhead_defect")
    part = Usd.Stage.CreateInMemory()
    plug = UsdGeom.Cylinder.Define(part, "/Part/Plug")
    plug.CreateRadiusAttr(0.0105)
    plug.CreateHeightAttr(plug_depth)
    plug.CreateAxisAttr(UsdGeom.Tokens.x)
    plug.AddTranslateOp().Set(Gf.Vec3d(plug_x, 0, 0.011))
    UsdPhysics.CollisionAPI.Apply(plug.GetPrim())
    plug.GetPrim().SetCustomDataByKey("arena:feature", "plug")
    _box(part, "/Part/Tab", (tab_x, 0, 0.015), (0.022, 0.020, 0.008), "pull_tab")
    assets = {"dust_cup": {"affordances": {"obstruction_socket": [0.056, socket_y, 0.026]}}}
    features = {"dust_cup": collision_features(cup), "obstruction": collision_features(part)}
    return assets, features


def test_obstruction_has_clear_bore_external_tab_and_supported_mass():
    from isaaclab_arena_environments.return_to_service.asset_source.fits import obstruction_inlet_fit

    assets, features = _obstruction_cell()
    report = obstruction_inlet_fit(assets, features)
    assert report["minimum_plug_cup_clearance_m"] == pytest.approx(0.0015, abs=1e-7)
    assert report["minimum_tab_cup_clearance_m"] == pytest.approx(0.002, abs=1e-7)
    assert report["axial_support_interval_m"] == pytest.approx((0.042, 0.068))
    assert report["estimated_com_support_margin_m"] > 0.0049


@pytest.mark.parametrize(
    "settings,message",
    (
        ({"plug_depth": 0.012, "plug_x": 0.0, "tab_x": 0.015}, "pull tab intersects cup"),
        ({"plug_depth": 0.012, "plug_x": 0.0}, "pull tab must remain connected"),
        ({"plug_depth": 0.014, "plug_x": 0.010}, "center of mass exceeds stable axial support"),
        ({"socket_y": 0.004}, "plug intersects cup"),
        ({"cup_intruder": True}, "plug intersects cup"),
    ),
)
def test_obstruction_collision_and_support_regressions_are_rejected(settings, message):
    from isaaclab_arena_environments.return_to_service.asset_source.fits import obstruction_inlet_fit

    assets, features = _obstruction_cell(**settings)
    with pytest.raises(AssertionError, match=message):
        obstruction_inlet_fit(assets, features)


def _box(stage, path, center, dimensions, feature):
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    if feature:
        box.GetPrim().SetCustomDataByKey("arena:feature", feature)
    return box


def _battery_channel(rail_center=0.034):
    from pxr import Usd

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features

    stage = Usd.Stage.CreateInMemory()
    for side, name in ((-1, "Left"), (1, "Right")):
        _box(stage, f"/Fixture/{name}", (0, side * rail_center, 0.03), (0.07, 0.005, 0.012), "load_rail")
    rails = collision_features(stage)["load_rail"]
    part_stage = Usd.Stage.CreateInMemory()
    _box(part_stage, "/Part/Housing", (0, 0, 0.018), (0.06, 0.055, 0.036), "housing")
    part = collision_features(part_stage)["housing"]
    return rails, part


def test_hidden_collision_shapes_retain_measured_assembly_clearance():
    from isaaclab_arena_environments.return_to_service.asset_source.fits import channel_clearances

    rails, parts = _battery_channel()
    assert channel_clearances(rails, parts[0].ComputeAlignedRange(), [0, 0, 0.025]) == pytest.approx((0.004, 0.004))


@pytest.mark.parametrize("rail_center,socket_y", ((0.03, 0), (0.034, 0.006)))
def test_narrowed_or_misaligned_channel_is_rejected(rail_center, socket_y):
    from isaaclab_arena_environments.return_to_service.asset_source.fits import validate_assembly_fits

    rails, parts = _battery_channel(rail_center)
    # An invalid first interface fails before unrelated task assets are required.
    assets = {"battery_tester": {"affordances": {"battery_socket": [0, socket_y, 0.025]}}}
    features = {"battery_tester": {"load_rail": rails}, "battery": {"housing": parts}}
    with pytest.raises(AssertionError, match="Insufficient assembly clearance: battery_tester/load_rail"):
        validate_assembly_fits(assets, features)


def test_collision_bounds_include_parent_transforms_and_primitive_scale():
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Asset")
    parent.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    parent.AddRotateZOp().Set(90)
    _box(stage, "/Asset/Rest", (0.1, 0, 0), (0.02, 0.04, 0.01), "neck_rest")
    bounds = collision_features(stage)["neck_rest"][0].ComputeAlignedRange()
    assert tuple(bounds.GetMin()) == pytest.approx((0.98, 2.09, 2.995))
    assert tuple(bounds.GetMax()) == pytest.approx((1.02, 2.11, 3.005))


def test_unlabeled_collision_primitive_is_rejected():
    from pxr import Usd

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features

    stage = Usd.Stage.CreateInMemory()
    _box(stage, "/Asset/Rest", (0, 0, 0), (0.02, 0.04, 0.01), None)
    with pytest.raises(AssertionError, match="Collision shape lacks feature name"):
        collision_features(stage)


def test_visual_geometry_cannot_substitute_for_collision_support():
    from pxr import Usd, UsdPhysics

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features

    stage = Usd.Stage.CreateInMemory()
    box = _box(stage, "/Asset/Rest", (0, 0, 0), (0.02, 0.04, 0.01), "neck_rest")
    box.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
    assert collision_features(stage) == {}


@pytest.mark.parametrize("changed_face", (None, (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)))
def test_tray_interior_matches_every_exported_collision_surface(changed_face):
    from pxr import Usd

    from isaaclab_arena_environments.return_to_service.asset_source.fits import collision_features, tray_interior_fit

    stage = Usd.Stage.CreateInMemory()
    _box(stage, "/Tray/Floor", (0, 0, 0.003), (0.15, 0.12, 0.006), "floor")
    for side in (-1, 1):
        suffix = "Negative" if side == -1 else "Positive"
        _box(stage, f"/Tray/X{suffix}", (side * 0.072, 0, 0.0325), (0.006, 0.12, 0.065), "wall_x")
        _box(stage, f"/Tray/Y{suffix}", (0, side * 0.057, 0.0325), (0.138, 0.006, 0.065), "wall_y")
    features = collision_features(stage)
    floor = features["floor"][0].ComputeAlignedRange()
    interior = [[-0.069, -0.054, 0.006], [0.069, 0.054, 0.065]]
    if changed_face is None:
        report = tray_interior_fit(floor, features["wall_x"], features["wall_y"], interior)
        assert report["collision_interior_bounds"][0] == pytest.approx(interior[0])
        assert report["collision_interior_bounds"][1] == pytest.approx(interior[1])
    else:
        corner, axis = changed_face
        interior[corner][axis] += 0.001
        with pytest.raises(AssertionError, match="interior face differs"):
            tray_interior_fit(floor, features["wall_x"], features["wall_y"], interior)


@pytest.mark.parametrize("rail_count", (0, 1, 3))
def test_channel_requires_two_physical_guides(rail_count):
    from isaaclab_arena_environments.return_to_service.asset_source.fits import channel_clearances

    rails, parts = _battery_channel()
    with pytest.raises(AssertionError, match="requires two collision rails"):
        channel_clearances([rails[0]] * rail_count, parts[0].ComputeAlignedRange(), [0, 0, 0.025])


@pytest.mark.parametrize("mount_z,mount_height,passes", ((0.025, 0.028, False), (0.021, 0.022, True)))
def test_exterior_latch_mount_cannot_enter_the_closing_lid(mount_z, mount_height, passes):
    from pxr import Usd

    from isaaclab_arena_environments.return_to_service.asset_source.fits import (
        collision_features,
        rotational_sweep_clearance,
    )

    stage = Usd.Stage.CreateInMemory()
    _box(stage, "/Case/LidFront", (0, -0.1285, 0.1025), (0.374, 0.008, 0.135), "lid_front")
    _box(stage, "/Case/Mount", (0.045, -0.141, mount_z), (0.018, 0.025, mount_height), "mount")
    features = collision_features(stage)
    if passes:
        assert (
            rotational_sweep_clearance(
                features["lid_front"], features["mount"], (0, 0.1325, 0.035), (1, 0, 0), (-110, 0), 0.002
            )
            > 0.002
        )
    else:
        with pytest.raises(AssertionError, match="Insufficient rotational clearance"):
            rotational_sweep_clearance(
                features["lid_front"], features["mount"], (0, 0.1325, 0.035), (1, 0, 0), (-110, 0), 0.002
            )


@pytest.mark.parametrize("perpendicular_offset,expected", ((0, -0.2), (0.2, 0), (0.25, 0.05)))
def test_oriented_box_separation_distinguishes_overlapping_axis_aligned_bounds(perpendicular_offset, expected):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import box_separation

    matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 45))
    bounds = Gf.Range3d(Gf.Vec3d(-1, -0.1, -0.1), Gf.Vec3d(1, 0.1, 0.1))
    box = Gf.BBox3d(bounds, matrix)
    offset = matrix.TransformDir(Gf.Vec3d(0, perpendicular_offset, 0))
    assert box_separation(box, box, offset) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("guide_height,interferes", ((0.019, True), (0.012, False)))
def test_filter_guide_corner_regression_is_invariant_under_rigid_frame_changes(guide_height, interferes):
    import math

    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import box_separation

    guide_half_size = Gf.Vec3d(0.035, 0.0015, guide_height / 2)
    guide = Gf.BBox3d(
        Gf.Range3d(-guide_half_size, guide_half_size), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.105, 0.0305, 0.062))
    )
    angle = math.pi / 8
    normal = Gf.Vec3d(0, math.cos(angle), math.sin(angle))
    tangent = Gf.Vec3d(0, -math.sin(angle), math.cos(angle))
    center = Gf.Vec3d(0.123, 0, 0.062) + 0.035 * normal
    wall_matrix = Gf.Matrix4d([*normal, 0], [*tangent, 0], [1, 0, 0, 0], [*center, 1])
    wall_half_size = Gf.Vec3d(0.002, 0.033 * math.tan(math.pi / 16), 0.045)
    wall = Gf.BBox3d(Gf.Range3d(-wall_half_size, wall_half_size), wall_matrix)
    separation = box_separation(guide, wall)
    assert (separation < 0) == interferes
    if not interferes:
        assert separation > 0.001
    transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 37))
    transform.SetTranslateOnly(Gf.Vec3d(0.4, -0.2, 0.8))
    moved_guide = Gf.BBox3d(guide.GetRange(), guide.GetMatrix() * transform)
    moved_wall = Gf.BBox3d(wall.GetRange(), wall.GetMatrix() * transform)
    assert box_separation(moved_guide, moved_wall) == pytest.approx(separation, abs=1e-10)


@pytest.mark.parametrize("display_y,display_depth,passes", ((0.039, 0.051, False), (0.045, 0.038, True)))
def test_cup_extraction_rejects_display_housing_that_pinches_the_sealing_lip(display_y, display_depth, passes):
    import math

    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import extraction_clearance

    segments = []
    half_size = Gf.Vec3d(0.003, 0.033 * math.tan(math.pi / 16), 0.003)
    for index in range(16):
        angle = math.tau * index / 16
        normal = Gf.Vec3d(0, math.sin(angle), -math.cos(angle))
        tangent = Gf.Vec3d(0, math.cos(angle), math.sin(angle))
        center = Gf.Vec3d(-0.042, 0, 0.037) + 0.036 * normal
        matrix = Gf.Matrix4d([*normal, 0], [*tangent, 0], [1, 0, 0, 0], [*center, 1])
        segments.append(Gf.BBox3d(Gf.Range3d(-half_size, half_size), matrix))
    display_half_size = Gf.Vec3d(0.0665, display_depth / 2, 0.013)
    display = Gf.BBox3d(
        Gf.Range3d(-display_half_size, display_half_size),
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, display_y, 0.031)),
    )
    if passes:
        assert extraction_clearance(segments, [display], (0.10, 0, 0), (0.087, 0.005, -0.040)) > 0.006
    else:
        with pytest.raises(AssertionError, match="Insufficient cup extraction clearance"):
            extraction_clearance(segments, [display], (0.10, 0, 0), (0.087, 0.005, -0.040))


def test_translated_sweep_preserves_a_rotated_scaled_box_and_contains_both_endpoints():
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import translated_sweep

    matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(0.2, 0.4, 0.6))
    matrix *= Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 2, 3), 37))
    matrix.SetTranslateOnly(Gf.Vec3d(0.3, -0.2, 0.1))
    box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.5), Gf.Vec3d(0.5)), matrix)
    displacement = Gf.Vec3d(0.1, -0.2, 0.3)
    swept = translated_sweep(box, displacement)
    assert swept.GetMatrix() == matrix
    for index in range(8):
        corner = box.GetRange().GetCorner(index)
        assert swept.GetRange().Contains(corner)
        moved = matrix.GetInverse().Transform(matrix.Transform(corner) + displacement)
        for axis in range(3):
            assert swept.GetRange().GetMin()[axis] - 1e-12 <= moved[axis]
            assert moved[axis] <= swept.GetRange().GetMax()[axis] + 1e-12


@pytest.mark.parametrize(
    "radial_offset,axial_offset,error", ((0, 0, None), (0.02, 0, "transverse"), (0, 0.002, "planes"))
)
def test_axial_seating_requires_both_coincident_planes_and_physical_overlap(radial_offset, axial_offset, error):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import axial_seating_fit

    seat = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.01, -0.005, -0.005), Gf.Vec3d(0, 0.005, 0.005)))
    lip = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(0, -0.005, -0.005), Gf.Vec3d(0.003, 0.005, 0.005)))
    socket = (axial_offset, radial_offset, 0)
    if error:
        with pytest.raises(AssertionError, match=error):
            axial_seating_fit([seat], [lip], socket)
    else:
        result = axial_seating_fit([seat], [lip], socket)
        assert result == {"axial_gap": 0.0, "contact_segment_count": 1}


@pytest.mark.parametrize("bore_radius,passes", ((0.033, False), (0.037, True)))
def test_cup_running_fit_rejects_the_original_pinching_bore(bore_radius, passes):
    import math

    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import extraction_clearance

    segments = []
    half_size = Gf.Vec3d(0.002, bore_radius * math.tan(math.pi / 16), 0.045)
    for index in range(16):
        angle = math.tau * index / 16
        normal = Gf.Vec3d(0, math.sin(angle), -math.cos(angle))
        tangent = Gf.Vec3d(0, math.cos(angle), math.sin(angle))
        center = Gf.Vec3d(0, 0, 0.037) + (bore_radius + 0.002) * normal
        matrix = Gf.Matrix4d([*normal, 0], [*tangent, 0], [1, 0, 0, 0], [*center, 1])
        segments.append(Gf.BBox3d(Gf.Range3d(-half_size, half_size), matrix))
    guide_half_size = Gf.Vec3d(0.035, 0.0015, 0.006)
    guide = Gf.BBox3d(
        Gf.Range3d(-guide_half_size, guide_half_size), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.105, 0.0305, 0.062))
    )
    args = (segments, [guide], (0.10, 0, 0), (-0.123, 0, -0.025))
    if passes:
        assert extraction_clearance(*args, minimum_required=0.003) >= 0.003
    else:
        with pytest.raises(AssertionError, match="Insufficient cup extraction clearance"):
            extraction_clearance(*args, minimum_required=0.003)


@pytest.mark.parametrize("offset,error", (((0, 0, 0), None), ((0, 0, 0.002), "height"), ((0, 0.002, 0), "beyond")))
def test_passive_skid_requires_supported_footprint_and_correct_height(offset, error):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import support_fit

    rail = Gf.Range3d(Gf.Vec3d(0.078, -0.010, 0.027), Gf.Vec3d(0.170, 0.010, 0.034))
    skid = Gf.Range3d(Gf.Vec3d(-0.033, -0.009, -0.006), Gf.Vec3d(0.037, 0.009, -0.0015))
    socket = [value + delta for value, delta in zip((0.123, 0, 0.040), offset)]
    if error:
        with pytest.raises(AssertionError, match=error):
            support_fit(rail, skid, socket, expected_gap=0)
    else:
        report = support_fit(rail, skid, socket, expected_gap=0)
        assert report["vertical_gap"] == pytest.approx(0, abs=1e-10)
        assert report["footprint_margins"] == pytest.approx((0.012, 0.010, 0.001, 0.001))


@pytest.mark.parametrize("rail_end,passes", ((0.178, False), (0.170, True)))
def test_cup_extraction_endpoint_clears_support_before_orientation_correction(rail_end, passes):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import extraction_clearance

    cup_lip = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.045, -0.009, -0.006), Gf.Vec3d(-0.039, 0.009, 0)))
    rail = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(0.078, -0.010, 0.027), Gf.Vec3d(rail_end, 0.010, 0.034)))
    args = ([cup_lip], [rail], (0, 0, 0), (-0.223, 0, -0.040))
    if passes:
        assert extraction_clearance(*args, minimum_required=0.006) == pytest.approx(0.008)
    else:
        with pytest.raises(AssertionError, match="extraction clearance"):
            extraction_clearance(*args, minimum_required=0.006)


@pytest.mark.parametrize(
    "catch_z,catch_y,error",
    ((0.066, -0.1365, None), (0.047, -0.1365, "gap"), (0.067, -0.1365, "gap"), (0.066, -0.11, "overlap")),
)
def test_case_hook_requires_a_close_catch_with_transverse_overlap(catch_z, catch_y, error):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import latch_capture_fit

    hook = Gf.Range3d(Gf.Vec3d(-0.0175, -0.1485, 0.0725), Gf.Vec3d(0.0175, -0.1335, 0.0795))
    center = Gf.Vec3d(0, catch_y, catch_z)
    half = Gf.Vec3d(0.018, 0.006, 0.006)
    catch = Gf.Range3d(center - half, center + half)
    if error:
        with pytest.raises(AssertionError, match=error):
            latch_capture_fit(hook, catch)
    else:
        result = latch_capture_fit(hook, catch)
        assert result["vertical_gap"] == pytest.approx(0.0005)
        assert result["overlaps_xy"] == pytest.approx((0.035, 0.009))


@pytest.mark.parametrize("obstacle_z,passes", ((0, False), (0.1, True)))
def test_rotation_bound_catches_collisions_between_separated_samples(obstacle_z, passes):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import (
        box_separation,
        rotational_sweep_clearance,
    )

    bounds = Gf.Range3d(Gf.Vec3d(-0.001), Gf.Vec3d(0.001))
    moving = Gf.BBox3d(bounds, Gf.Matrix4d().SetTranslate(Gf.Vec3d(1, 0, 0)))
    obstacle = Gf.BBox3d(bounds, Gf.Matrix4d().SetTranslate(Gf.Vec3d(1, 0, obstacle_z)))
    # Both sampled endpoints are clear even when the intervening arc collides.
    for angle in (-1, 1):
        rotated = Gf.BBox3d(bounds, moving.GetMatrix() * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), angle)))
        assert box_separation(rotated, obstacle) > 0
    if passes:
        assert (
            rotational_sweep_clearance([moving], [obstacle], (0, 0, 0), (0, 0, 1), (-1, 1), max_step_degrees=2) > 0.08
        )
    else:
        with pytest.raises(AssertionError, match="rotational clearance"):
            rotational_sweep_clearance([moving], [obstacle], (0, 0, 0), (0, 0, 1), (-1, 1), max_step_degrees=2)


@pytest.mark.parametrize("solid_keeper", (True, False))
def test_latch_sweep_rejects_a_mount_that_intersects_the_lever(solid_keeper):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import rotational_sweep_clearance

    lever = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.018, -0.014, 0), Gf.Vec3d(0.018, -0.004, 0.052)))
    if solid_keeper:
        keeper = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.024, -0.001, -0.0075), Gf.Vec3d(0.024, 0.009, 0.0075)))
        with pytest.raises(AssertionError, match="rotational clearance"):
            rotational_sweep_clearance([lever], [keeper], (0, 0, 0), (1, 0, 0), (0, 100))
    else:
        ear = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(0.020, -0.003, -0.0075), Gf.Vec3d(0.027, 0.009, 0.0075)))
        assert rotational_sweep_clearance([lever], [ear], (0, 0, 0), (1, 0, 0), (0, 100)) > 0.0018


@pytest.mark.parametrize("pin_radius", (0.0045, 0.0065))
def test_case_sweep_checks_the_pin_against_every_part_except_its_mounting_ears(pin_radius):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import _case_latch_fits

    def box(center, size):
        half = Gf.Vec3d(*size) / 2
        return Gf.BBox3d(Gf.Range3d(-half, half), Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center)))

    assets = {
        "case_base": {
            "affordances": {
                "latch": {"position_xyz": [0, -0.14, 0.029], "axis": [1, 0, 0], "angle_limits_degrees": [0, 100]},
                "hinge": {"axis": [1, 0, 0], "angle_limits_degrees": [-110, 0]},
                "lid_closed_pose": {"position_xyz": [0, 0.1325, 0.035]},
            }
        }
    }
    features = {
        "case_base": {
            "latch_keeper": [box((0, -0.1325, 0.029), (0.054, 0.003, 0.015))],
            "latch_pivot_ear": [box((0.0235, -0.137, 0.029), (0.007, 0.012, 0.015))],
        },
        "case_lid": {
            "catch": [box((0, -0.269, 0.031), (0.036, 0.012, 0.012))],
            "lid_end": [box((0, -0.261, 0.0675), (0.374, 0.008, 0.135))],
        },
        "case_latch": {
            "pivot": [box((0, 0, 0), (0.054, 2 * pin_radius, 2 * pin_radius))],
            "lever": [box((0, -0.009, 0.026), (0.036, 0.010, 0.052))],
            "hook": [box((0, -0.001, 0.047), (0.035, 0.015, 0.007))],
            "grasp_flange": [box((0, -0.018, 0.020), (0.046, 0.015, 0.018))],
        },
    }
    if pin_radius > 0.005:
        with pytest.raises(AssertionError, match="latch-pin clearance"):
            _case_latch_fits(assets, features)
    else:
        report = _case_latch_fits(assets, features)
        assert report["pivot_mount_clearance"] == pytest.approx(0.0015)
        assert report["latch_sweep_clearance"] > 0.0003
        assert report["lid_sweep_clearance"] > 0.0023


def test_translation_volume_checks_simultaneous_offsets_between_safe_endpoints():
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import (
        box_separation,
        translation_volume_clearance,
    )

    shape = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(-0.01), Gf.Vec3d(0.01)))
    limits = ((-0.1, -0.1, 0), (0.1, 0.1, 0))
    for endpoint in limits:
        assert box_separation(shape, shape, endpoint) > 0
    with pytest.raises(AssertionError, match="translation-volume clearance"):
        translation_volume_clearance([shape], [shape], limits)


@pytest.mark.parametrize(
    "height,footprint_error,error", ((0.001, 0, None), (0.003, 0, "clearance"), (0.001, 0.001, "footprint"))
)
def test_keyed_filter_guide_preserves_ring_travel_and_measured_footprint(height, footprint_error, error):
    import math

    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import _filter_guide_fits

    def box(center, size):
        half = Gf.Vec3d(*size) / 2
        return Gf.BBox3d(Gf.Range3d(-half, half), Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center)))

    caps = []
    half = Gf.Vec3d(0.006, 0.014 * math.tan(math.pi / 16), 0.0025)
    for x in (-0.025, 0.025):
        for index in range(16):
            angle = math.tau * index / 16
            normal = Gf.Vec3d(0, math.sin(angle), -math.cos(angle))
            tangent = Gf.Vec3d(0, math.cos(angle), math.sin(angle))
            center = Gf.Vec3d(x, 0, 0.026) + 0.020 * normal
            matrix = Gf.Matrix4d([*normal, 0], [*tangent, 0], [1, 0, 0, 0], [*center, 1])
            caps.append(Gf.BBox3d(Gf.Range3d(-half, half), matrix))
    features = {
        "vacuum_body": {
            "filter_lower_guide": [box((0.105, 0, 0.0335), (0.070, 0.022, 0.004))],
            "filter_skid_kerb": [
                box((0.105, side * 0.0105, 0.0355 + height / 2), (0.070, 0.001, height)) for side in (-1, 1)
            ],
        },
        "filter": {"anti_roll_skid": [box((0, 0, 0.0015), (0.054, 0.018, 0.003))], "end_cap": caps},
        "filter_decoy": {"anti_roll_skid": [box((0, 0, -0.0025), (0.054, 0.018, 0.003))]},
        "dust_cup": {"tab": [box((0, 0, 0.079), (0.023, 0.019, 0.012))]},
    }
    assets = {
        "vacuum_body": {
            "affordances": {
                "filter_socket": [0.103, 0, 0.036],
                "cup_socket": [0.123, 0, 0.025],
                "filter_capture_bounds": [[0.0705, -0.0099, 0.03545], [0.1395, 0.0099, 0.0392]],
            }
        },
        "filter": {
            "affordances": {"support_footprint_bounds": [[-0.027, -0.009 + footprint_error, 0], [0.027, 0.009, 0.003]]}
        },
        "filter_decoy": {
            "affordances": {"support_footprint_bounds": [[-0.027, -0.009, -0.004], [0.027, 0.009, -0.001]]}
        },
    }
    if error:
        with pytest.raises(AssertionError, match=error):
            _filter_guide_fits(assets, features)
    else:
        result = _filter_guide_fits(assets, features)
        assert result["side_clearances"] == pytest.approx((0.001, 0.001))
        assert result["nominal_vertical_engagement"] == pytest.approx((0.0005, 0.0005))
        assert result["settled_origin_z"] == pytest.approx(0.0355)
        assert result["filter_running_clearance"] > 0.00054
        assert result["skid_running_clearance"] == pytest.approx(0)
        assert result["extracted_endpoint_clearance"] == pytest.approx(0.0105)


@pytest.mark.parametrize(
    "mutation,error", ((None, None), ("wide", "side margin"), ("raised", "engagement"), ("high_floor", "excludes"))
)
def test_filter_capture_metadata_cannot_drift_away_from_physical_guidance(mutation, error):
    from pxr import Gf

    from isaaclab_arena_environments.return_to_service.asset_source.fits import filter_capture_fit

    support = Gf.Range3d(Gf.Vec3d(0.070, -0.011, 0.0315), Gf.Vec3d(0.140, 0.011, 0.0355))
    skid = Gf.Range3d(Gf.Vec3d(-0.027, -0.009, 0), Gf.Vec3d(0.027, 0.009, 0.003))
    kerbs = [
        Gf.BBox3d(Gf.Range3d(Gf.Vec3d(0.070, -0.011, 0.0355), Gf.Vec3d(0.140, -0.010, 0.0365))),
        Gf.BBox3d(Gf.Range3d(Gf.Vec3d(0.070, 0.010, 0.0355), Gf.Vec3d(0.140, 0.011, 0.0365))),
    ]
    capture = [[0.0705, -0.0099, 0.03545], [0.1395, 0.0099, 0.0392]]
    if mutation == "wide":
        capture[1][1] += 0.0002
    elif mutation == "raised":
        capture[1][2] += 0.001
    elif mutation == "high_floor":
        capture[0][2] += 0.0001
    if error:
        with pytest.raises(AssertionError, match=error):
            filter_capture_fit(support, kerbs, skid, (0.103, 0, 0.036), capture)
    else:
        result = filter_capture_fit(support, kerbs, skid, (0.103, 0, 0.036), capture)
        assert result["horizontal_margins"] == pytest.approx((0.000499, 0.000499, 0.000099, 0.000099))
        assert result["minimum_vertical_engagement"] > 0.0002875
