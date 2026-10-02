# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure task-specific assembly clearances from exported collision primitives."""

from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Float32 USD transforms may move a nominal interface by a few nanometers.
FIT_TOLERANCE = 1e-7


def collision_features(stage: Usd.Stage) -> dict[str, list[Gf.BBox3d]]:
    """Collect actual primitive bounds by authored manufacturing-feature name.

    Args:
        stage: One exported part, with identity root transform and meter units.

    Returns:
        Feature names mapped to primitive bounds in the part's local frame.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
    features = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        feature = prim.GetCustomDataByKey("arena:feature")
        assert isinstance(feature, str) and feature, f"Collision shape lacks feature name: {prim.GetPath()}"
        bounds = cache.ComputeWorldBound(prim)
        assert not bounds.GetRange().IsEmpty(), f"Empty collision bounds: {prim.GetPath()}"
        features.setdefault(feature, []).append(bounds)
    return features


def _feature_bounds(features: dict, asset: str, feature: str) -> Gf.Range3d:
    """Combine all exported collision segments of a named manufacturing feature."""
    shapes = features[asset].get(feature, [])
    assert shapes, f"Missing collision feature: {asset}/{feature}"
    bounds = Gf.Range3d()
    for shape in shapes:
        bounds.UnionWith(shape.ComputeAlignedRange())
    return bounds


def channel_clearances(rails: list[Gf.BBox3d], part: Gf.Range3d, socket: list[float]) -> tuple[float, float]:
    """Measure both Y clearances for a part translated into a parallel guide channel.

    Args:
        rails: Two exported rail bounds in the receiving fixture frame.
        part: Exported mating-part bounds in its local frame.
        socket: Part origin in fixture coordinates, with unchanged orientation.

    Returns:
        Clearance at the negative and positive Y rail respectively, in meters.
    """
    assert len(rails) == 2, "Guide channel requires two collision rails"
    aligned = [rail.ComputeAlignedRange() for rail in rails]
    lower, upper = sorted(aligned, key=lambda bounds: bounds.GetMin()[1])
    return (
        part.GetMin()[1] + socket[1] - lower.GetMax()[1],
        upper.GetMin()[1] - part.GetMax()[1] - socket[1],
    )


def box_separation(left: Gf.BBox3d, right: Gf.BBox3d, right_offset=(0, 0, 0)) -> float:
    """Bound oriented-box separation using face normals and edge cross products.

    Args:
        left: First collision box in the reference frame.
        right: Second collision box, with a rigid or orthogonal-scale transform.
        right_offset: Translation placing the second part in the reference frame.

    Returns:
        Positive lower bound on distance, or a nonpositive value for touching or overlapping boxes.
    """
    centers, half_axes = [], []
    for box in (left, right):
        matrix = box.GetMatrix()
        bounds = box.GetRange()
        centers.append(matrix.Transform(bounds.GetMidpoint()))
        half_size = bounds.GetSize() / 2
        vectors = []
        for axis in range(3):
            vector = Gf.Vec3d(0)
            vector[axis] = half_size[axis]
            vectors.append(matrix.TransformDir(vector))
        half_axes.append(vectors)
    delta = centers[1] + Gf.Vec3d(*right_offset) - centers[0]
    axes = half_axes[0] + half_axes[1]
    for left_axis in half_axes[0]:
        for right_axis in half_axes[1]:
            axes.append(Gf.Cross(left_axis.GetNormalized(), right_axis.GetNormalized()))
    separation = -float("inf")
    for axis in axes:
        if axis.GetLength() < 1e-12:
            continue
        normal = axis.GetNormalized()
        radius = sum(abs(Gf.Dot(normal, vector)) for vector in half_axes[0] + half_axes[1])
        separation = max(separation, abs(Gf.Dot(normal, delta)) - radius)
    return separation


def axial_cylinder_box_separation(cylinder: Gf.Range3d, box: Gf.BBox3d, cylinder_offset) -> float:
    """Bound separation of an X-axis cylinder and a box with one X-aligned axis.

    Args:
        cylinder: Bounds of a circular X-axis cylinder in its local frame.
        box: Exported collision box in the receiving frame.
        cylinder_offset: Translation placing the cylinder in the receiving frame.

    Returns:
        Positive separation, zero contact, or a negative penetration bound.
        Circular cross sections preserve the open space inside segmented bores.
    """
    dimensions = cylinder.GetSize()
    assert abs(dimensions[1] - dimensions[2]) <= FIT_TOLERANCE, "Cylinder cross section must be circular"
    center = cylinder.GetMidpoint() + Gf.Vec3d(*cylinder_offset)
    matrix = box.GetMatrix()
    delta = center - matrix.Transform(box.GetRange().GetMidpoint())
    half = box.GetRange().GetSize() / 2
    radial_distance_squared = 0.0
    axial_separation = None
    for axis in range(3):
        direction = Gf.Vec3d(0)
        direction[axis] = 1
        vector = matrix.TransformDir(direction)
        normal = vector.GetNormalized()
        distance = abs(Gf.Dot(normal, delta)) - half[axis] * vector.GetLength()
        if abs(normal[0]) >= 1 - 1e-6:
            assert axial_separation is None, "A box must have exactly one axial direction"
            axial_separation = distance - dimensions[0] / 2
        else:
            assert abs(normal[0]) <= 1e-6, "Cylinder fit requires X-extruded collision boxes"
            radial_distance_squared += max(distance, 0.0) ** 2
    assert axial_separation is not None, "Cylinder fit requires an X-aligned collision-box axis"
    return max(axial_separation, math.sqrt(radial_distance_squared) - dimensions[1] / 2)


def _obstruction_support(plug: Gf.Range3d, tab: Gf.Range3d, inlet: Gf.Range3d, socket) -> dict:
    """Check connected geometry and a static support margin using uniform collider volumes."""
    for axis in range(3):
        overlap = min(plug.GetMax()[axis], tab.GetMax()[axis]) - max(plug.GetMin()[axis], tab.GetMin()[axis])
        assert overlap >= 0.001 - FIT_TOLERANCE, "Obstruction pull tab must remain connected to its plug"
    support = (
        max(plug.GetMin()[0] + socket[0], inlet.GetMin()[0]),
        min(plug.GetMax()[0] + socket[0], inlet.GetMax()[0]),
    )
    plug_size, tab_size = plug.GetSize(), tab.GetSize()
    plug_volume = math.pi * (plug_size[1] / 2) ** 2 * plug_size[0]
    tab_volume = tab_size[0] * tab_size[1] * tab_size[2]
    center_x = (
        (plug.GetMidpoint()[0] * plug_volume + tab.GetMidpoint()[0] * tab_volume) / (plug_volume + tab_volume)
    ) + socket[0]
    margin = min(center_x - support[0], support[1] - center_x)
    assert margin >= 0.003 - FIT_TOLERANCE, "Obstruction estimated center of mass exceeds stable axial support"
    return {
        "axial_support_interval_m": list(support),
        "estimated_com_x_m": center_x,
        "estimated_com_support_margin_m": margin,
        "minimum_com_support_margin_m": 0.003,
        "mass_estimate": "Uniform density over the cylinder and box collision primitives; static screening only.",
    }


def obstruction_inlet_fit(assets: dict, features: dict) -> dict:
    """Reject colliding or unsupported obstruction geometry using the exported cup and plug.

    Args:
        assets: Blender export manifest records containing the initial obstruction socket.
        features: Actual exported collision bounds collected with collision_features.

    Returns:
        Measured clearances and a static support estimate. Physical settling and
        deliberate extraction still require simulation validation.
    """
    socket = assets["dust_cup"]["affordances"]["obstruction_socket"]
    obstruction = features["obstruction"]
    assert len(obstruction["plug"]) == len(obstruction["pull_tab"]) == 1
    plug = _feature_bounds(features, "obstruction", "plug")
    tab = _feature_bounds(features, "obstruction", "pull_tab")
    inlet = _feature_bounds(features, "dust_cup", "inlet")
    minimum_plug, minimum_tab = float("inf"), float("inf")
    for shapes in features["dust_cup"].values():
        for shape in shapes:
            minimum_plug = min(minimum_plug, axial_cylinder_box_separation(plug, shape, socket))
            minimum_tab = min(minimum_tab, box_separation(shape, obstruction["pull_tab"][0], socket))
    assert (
        minimum_tab >= 0.001 - FIT_TOLERANCE
    ), f"Obstruction pull tab intersects cup or lacks clearance: {minimum_tab}"
    assert minimum_plug >= 0.001 - FIT_TOLERANCE, f"Obstruction plug intersects cup or lacks clearance: {minimum_plug}"
    return {
        "minimum_plug_cup_clearance_m": minimum_plug,
        "minimum_tab_cup_clearance_m": minimum_tab,
        **_obstruction_support(plug, tab, inlet, socket),
    }


def translated_sweep(box: Gf.BBox3d, displacement) -> Gf.BBox3d:
    """Bound a collision box's continuous translation without changing orientation.

    Args:
        box: Starting collision bound in the reference frame.
        displacement: End minus start position in the reference frame.

    Returns:
        Conservative swept bound, exact when translation follows a box axis.
    """
    delta = box.GetMatrix().GetInverse().TransformDir(Gf.Vec3d(*displacement))
    lower, upper = Gf.Vec3d(box.GetRange().GetMin()), Gf.Vec3d(box.GetRange().GetMax())
    for axis in range(3):
        lower[axis] += min(delta[axis], 0)
        upper[axis] += max(delta[axis], 0)
    return Gf.BBox3d(Gf.Range3d(lower, upper), box.GetMatrix())


def extraction_clearance(
    moving: list[Gf.BBox3d], obstacles: list[Gf.BBox3d], displacement, obstacle_offset, minimum_required: float = 0.005
) -> float:
    """Check a continuous extraction corridor against nearby fixture geometry.

    Args:
        moving: Exported part collision boxes in its starting frame.
        obstacles: Fixed fixture collision boxes in their asset frame.
        displacement: Translation of the extracted part in its starting frame.
        obstacle_offset: Fixture origin relative to the part's starting frame, with aligned asset axes.
        minimum_required: Smallest permitted separation in meters.

    Returns:
        Conservative minimum separation in meters, meeting the required clearance.
    """
    assert moving and obstacles, "Extraction clearance requires both collision geometries"
    minimum = float("inf")
    for shape in moving:
        swept = translated_sweep(shape, displacement)
        for obstacle in obstacles:
            minimum = min(minimum, box_separation(swept, obstacle, obstacle_offset))
    assert minimum >= minimum_required - FIT_TOLERANCE, f"Insufficient cup extraction clearance: {minimum}"
    return minimum


def translation_volume_clearance(
    moving: list[Gf.BBox3d], obstacles: list[Gf.BBox3d], origin_bounds, minimum_required: float = 0.0
) -> float:
    """Certify separation while a part's origin traverses a rectangular volume.

    Args:
        moving: Collision boxes in the moving part's local frame.
        obstacles: Fixed collision boxes in the receiving fixture frame.
        origin_bounds: Lower and upper origin coordinates in the fixture frame, with aligned part axes.
        minimum_required: Smallest permitted separation in meters.

    Returns:
        Conservative separation over all simultaneous translations in the volume.
    """
    assert moving and obstacles, "Translation clearance requires both collision geometries"
    lower, upper = (Gf.Vec3d(*corner) for corner in origin_bounds)
    assert all(lower[axis] <= upper[axis] for axis in range(3)), "Invalid translation bounds"
    transform = Gf.Matrix4d().SetTranslate(lower)
    minimum = float("inf")
    for shape in moving:
        swept = Gf.BBox3d(shape.GetRange(), shape.GetMatrix() * transform)
        for axis in range(3):
            displacement = Gf.Vec3d(0)
            displacement[axis] = upper[axis] - lower[axis]
            swept = translated_sweep(swept, displacement)
        for obstacle in obstacles:
            minimum = min(minimum, box_separation(swept, obstacle))
    assert minimum >= minimum_required - FIT_TOLERANCE, f"Insufficient translation-volume clearance: {minimum}"
    return minimum


def axial_seating_fit(seat: list[Gf.BBox3d], lip: list[Gf.BBox3d], socket, axis: int = 0) -> dict:
    """Check a segmented shoulder physically stops an inserted part at its socket.

    Args:
        seat: Receiving shoulder collision boxes in the fixture frame.
        lip: Mating lip collision boxes in the inserted part's frame.
        socket: Inserted part origin in the fixture frame, with aligned axes.
        axis: Positive extraction axis; insertion moves in the negative direction.

    Returns:
        Axial gap and number of lip segments with physical seating contact.
    """
    assert seat and lip, "Axial seating requires both collision geometries"
    front = max(shape.ComputeAlignedRange().GetMax()[axis] for shape in seat)
    gaps = [shape.ComputeAlignedRange().GetMin()[axis] + socket[axis] - front for shape in lip]
    assert max(abs(gap) for gap in gaps) <= FIT_TOLERANCE, f"Seating planes do not coincide: {gaps}"
    # Coincident planes alone do not prove contact between two hollow rings.
    # A tiny overtravel must intersect every lip segment with the shoulder.
    overtravel = 0.0001
    pressed = list(socket)
    pressed[axis] -= overtravel
    for segment in lip:
        nominal = min(box_separation(shape, segment, socket) for shape in seat)
        assert nominal >= -FIT_TOLERANCE, f"Seating lip penetrates shoulder: {nominal}"
        compressed = min(box_separation(shape, segment, pressed) for shape in seat)
        assert compressed <= -overtravel + FIT_TOLERANCE, "Seating lip lacks transverse shoulder overlap"
    return {"axial_gap": min(gaps), "contact_segment_count": len(lip)}


def rotational_sweep_clearance(
    moving: list[Gf.BBox3d],
    obstacles: list[Gf.BBox3d],
    pivot,
    axis,
    angle_range_degrees,
    minimum_required: float = 0.0,
    max_step_degrees: float = 0.25,
) -> float:
    """Certify continuous hinged travel using box separation and a vertex-motion bound.

    Args:
        moving: Collision boxes in the reference frame before rotation.
        obstacles: Fixed collision boxes in the same reference frame.
        pivot: Rotation center in the reference frame.
        axis: Nonzero rotation axis in the reference frame.
        angle_range_degrees: Start and end rotation angles, in degrees.
        minimum_required: Smallest permitted separation in meters.
        max_step_degrees: Maximum sampling interval, strictly between zero and 180.

    Returns:
        Conservative separation over the entire angular interval, in meters.
    """
    assert moving and obstacles, "Rotational clearance requires both collision geometries"
    assert 0 < max_step_degrees < 180, "Invalid angular sampling interval"
    direction = Gf.Vec3d(*axis)
    assert direction.GetLength() > 0, "Rotation axis must be nonzero"
    direction.Normalize()
    center = Gf.Vec3d(*pivot)
    start, end = angle_range_degrees
    intervals = max(1, math.ceil(abs(end - start) / max_step_degrees))
    step = abs(end - start) / intervals
    rotations = []
    for index in range(intervals + 1):
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(direction, start + (end - start) * index / intervals))
        rotations.append(Gf.Matrix4d().SetTranslate(-center) * rotation * Gf.Matrix4d().SetTranslate(center))
    minimum = float("inf")
    for shape in moving:
        radius = 0.0
        for index in range(8):
            vertex = shape.GetMatrix().Transform(shape.GetRange().GetCorner(index))
            radius = max(radius, Gf.Cross(vertex - center, direction).GetLength())
        # Every intervening angle lies within half an interval of a sample.
        motion_bound = 2 * radius * math.sin(math.radians(step) / 4)
        for rotation in rotations:
            rotated = Gf.BBox3d(shape.GetRange(), shape.GetMatrix() * rotation)
            for obstacle in obstacles:
                minimum = min(minimum, box_separation(rotated, obstacle) - motion_bound)
    assert minimum >= minimum_required - FIT_TOLERANCE, f"Insufficient rotational clearance: {minimum}"
    return minimum


def latch_capture_fit(hook: Gf.Range3d, catch: Gf.Range3d) -> dict:
    """Check that a closed hook overlaps its catch with a small positive vertical gap.

    Args:
        hook: Closed hook's axis-aligned bounds in the case frame.
        catch: Closed lid catch's axis-aligned bounds in the case frame.

    Returns:
        Vertical gap and transverse overlaps, in meters.
    """
    gap = hook.GetMin()[2] - catch.GetMax()[2]
    assert -FIT_TOLERANCE <= gap <= 0.001 + FIT_TOLERANCE, f"Invalid hook-to-catch gap: {gap}"
    overlaps = []
    for axis in (0, 1):
        overlaps.append(min(hook.GetMax()[axis], catch.GetMax()[axis]) - max(hook.GetMin()[axis], catch.GetMin()[axis]))
    assert min(overlaps) >= 0.008 - FIT_TOLERANCE, f"Insufficient hook-to-catch overlap: {overlaps}"
    return {"vertical_gap": gap, "overlaps_xy": overlaps}


def tray_interior_fit(
    floor: Gf.Range3d,
    walls_x: list[Gf.BBox3d],
    walls_y: list[Gf.BBox3d],
    interior_bounds: list[list[float]],
) -> dict:
    """Require the declared tray interior to match its physical floor and walls.

    Args:
        floor: Exported collision floor bounds in the tray frame.
        walls_x: Two exported collision walls bounding the tray's X coordinates.
        walls_y: Two exported collision walls bounding the tray's Y coordinates.
        interior_bounds: Lower and upper declared interior corners in that frame.

    Returns:
        Measured interior bounds in meters, derived from collision surfaces.
    """
    lower, upper, rim_heights = [], [], []
    for axis, walls in enumerate((walls_x, walls_y)):
        assert len(walls) == 2, "Tray requires two opposing walls per horizontal axis"
        bounds = sorted((wall.ComputeAlignedRange() for wall in walls), key=lambda item: item.GetMin()[axis])
        lower.append(bounds[0].GetMax()[axis])
        upper.append(bounds[1].GetMin()[axis])
        rim_heights.extend(wall.GetMax()[2] for wall in bounds)
    assert max(rim_heights) - min(rim_heights) <= FIT_TOLERANCE, "Tray walls must share one opening height"
    lower.append(floor.GetMax()[2])
    upper.append(min(rim_heights))
    measured = [lower, upper]
    for corner in range(2):
        for axis in range(3):
            assert (
                abs(interior_bounds[corner][axis] - measured[corner][axis]) <= FIT_TOLERANCE
            ), f"Tray interior face differs from collision surface: corner={corner}, axis={axis}"
    return {"collision_interior_bounds": measured}


def _case_latch_fits(assets: dict, features: dict) -> dict:
    """Measure capture and continuous latch/lid travel from exported case geometry."""
    case = assets["case_base"]["affordances"]
    pivot = case["latch"]["position_xyz"]
    latch_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pivot))
    lid_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*case["lid_closed_pose"]["position_xyz"]))
    lid, base, latch, pin_obstacles = [], [], [], []
    for asset, transform, destination in (
        ("case_base", Gf.Matrix4d(1), base),
        ("case_lid", lid_transform, lid),
        ("case_latch", latch_transform, latch),
    ):
        for feature, shapes in features[asset].items():
            # The pin's mounting contact with its supporting ears is intended.
            if asset == "case_latch" and feature == "pivot":
                continue
            for shape in shapes:
                placed = Gf.BBox3d(shape.GetRange(), shape.GetMatrix() * transform)
                destination.append(placed)
                if asset != "case_latch" and not (asset == "case_base" and feature == "latch_pivot_ear"):
                    pin_obstacles.append(placed)
    hook = features["case_latch"]["hook"][0]
    catch = features["case_lid"]["catch"][0]
    report = latch_capture_fit(
        Gf.BBox3d(hook.GetRange(), hook.GetMatrix() * latch_transform).ComputeAlignedRange(),
        Gf.BBox3d(catch.GetRange(), catch.GetMatrix() * lid_transform).ComputeAlignedRange(),
    )
    # The coaxial cylinder is invariant under latch rotation; its containing
    # box remains fixed. Only contact with the supporting ears is intentional.
    pin = features["case_latch"]["pivot"][0]
    placed_pin = Gf.BBox3d(pin.GetRange(), pin.GetMatrix() * latch_transform)
    pin_clearance = min(box_separation(placed_pin, obstacle) for obstacle in pin_obstacles)
    assert pin_clearance >= 0.001 - FIT_TOLERANCE, f"Insufficient latch-pin clearance: {pin_clearance}"
    report["pivot_mount_clearance"] = pin_clearance
    report["latch_sweep_clearance"] = rotational_sweep_clearance(
        latch,
        base + lid,
        pivot,
        case["latch"]["axis"],
        case["latch"]["angle_limits_degrees"],
        minimum_required=0.0002,
    )
    opened = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(*case["latch"]["axis"]), case["latch"]["angle_limits_degrees"][1])
    )
    released = []
    for feature, shapes in features["case_latch"].items():
        for shape in shapes:
            rotation = Gf.Matrix4d(1) if feature == "pivot" else opened
            released.append(Gf.BBox3d(shape.GetRange(), shape.GetMatrix() * rotation * latch_transform))
    report["lid_sweep_clearance"] = rotational_sweep_clearance(
        lid,
        released,
        case["lid_closed_pose"]["position_xyz"],
        case["hinge"]["axis"],
        case["hinge"]["angle_limits_degrees"],
        minimum_required=0.002,
    )
    exterior_mount = []
    for feature, shapes in features["case_base"].items():
        if feature.startswith("latch_mount_"):
            exterior_mount.extend(shapes)
    if exterior_mount:
        exterior_mount.extend(features["case_base"]["latch_pivot_ear"])
        report["lid_mount_sweep_clearance"] = rotational_sweep_clearance(
            lid,
            exterior_mount,
            case["lid_closed_pose"]["position_xyz"],
            case["hinge"]["axis"],
            case["hinge"]["angle_limits_degrees"],
            minimum_required=0.002,
        )
    return report


def support_fit(support: Gf.Range3d, skid: Gf.Range3d, socket, expected_gap: float) -> dict:
    """Check a flat skid's height and full footprint on a rectangular support.

    Args:
        support: Axis-aligned collision bounds in the receiving fixture frame.
        skid: Axis-aligned collision bounds in the supported part frame.
        socket: Part origin in fixture coordinates, with unchanged orientation.
        expected_gap: Designed vertical clearance before gravity settles the part.

    Returns:
        Vertical gap and horizontal containment margins, in meters.
    """
    gap = skid.GetMin()[2] + socket[2] - support.GetMax()[2]
    assert abs(gap - expected_gap) <= FIT_TOLERANCE, f"Changed skid support height: {gap}"
    margins = []
    for axis in (0, 1):
        margins.extend((
            skid.GetMin()[axis] + socket[axis] - support.GetMin()[axis],
            support.GetMax()[axis] - skid.GetMax()[axis] - socket[axis],
        ))
    assert min(margins) >= -FIT_TOLERANCE, f"Skid extends beyond its support: {margins}"
    return {"vertical_gap": gap, "footprint_margins": margins}


def filter_capture_fit(support: Gf.Range3d, kerbs: list[Gf.BBox3d], skid: Gf.Range3d, socket, capture_bounds) -> dict:
    """Check capture metadata against physical support and the five-degree socket gate.

    Args:
        support: Lower guide bounds in the receiving body frame.
        kerbs: Two guide kerbs in the body frame.
        skid: Filter skid bounds in the filter frame.
        socket: Nominal filter origin in the body frame, with aligned axes.
        capture_bounds: Body-frame minimum and maximum seating-volume coordinates.

    Returns:
        Physical margins including the runtime's one-micrometer containment tolerance.
    """
    assert len(kerbs) == 2, "Filter capture requires two kerbs"
    lower_kerb, upper_kerb = sorted(
        (kerb.ComputeAlignedRange() for kerb in kerbs), key=lambda bounds: bounds.GetMin()[1]
    )
    # Keep these allowances synchronized with box_contained and Socket's angle gate.
    containment_tolerance = 0.000001
    lower = Gf.Vec3d(*capture_bounds[0]) - Gf.Vec3d(containment_tolerance)
    upper = Gf.Vec3d(*capture_bounds[1]) + Gf.Vec3d(containment_tolerance)
    margins = [
        lower[0] - support.GetMin()[0],
        support.GetMax()[0] - upper[0],
        lower[1] - lower_kerb.GetMax()[1],
        upper_kerb.GetMin()[1] - upper[1],
    ]
    assert min(margins[:2]) >= 0.000499 - FIT_TOLERANCE, "Filter capture extends beyond guide end margin"
    assert min(margins[2:]) >= 0.000099 - FIT_TOLERANCE, "Filter capture extends beyond kerb side margin"
    penetration = support.GetMax()[2] - lower[2]
    assert penetration <= 0.000051 + FIT_TOLERANCE, "Filter capture permits excessive support penetration"
    top = min(lower_kerb.GetMax()[2], upper_kerb.GetMax()[2])
    engagement = top - upper[2] + skid.GetSize()[2] * math.cos(math.radians(5))
    assert engagement >= 0.00028 - FIT_TOLERANCE, "Filter capture permits loss of kerb engagement"
    for height in (socket[2], support.GetMax()[2] - skid.GetMin()[2]):
        position = Gf.Vec3d(socket[0], socket[1], height)
        for axis in range(3):
            assert (
                skid.GetMin()[axis] + position[axis] >= lower[axis] - FIT_TOLERANCE
            ), "Filter capture excludes a nominal or settled skid"
            assert (
                skid.GetMax()[axis] + position[axis] <= upper[axis] + FIT_TOLERANCE
            ), "Filter capture excludes a nominal or settled skid"
    return {
        "horizontal_margins": margins,
        "maximum_support_penetration": penetration,
        "minimum_vertical_engagement": engagement,
        "containment_tolerance": containment_tolerance,
        "maximum_socket_angle_degrees": 5,
    }


def _filter_guide_fits(assets: dict, features: dict) -> dict:
    """Check keyed skid guidance across settlement, axial travel and cup prelift."""
    body = assets["vacuum_body"]["affordances"]
    socket = body["filter_socket"]
    kerbs = features["vacuum_body"]["filter_skid_kerb"]
    support = _feature_bounds(features, "vacuum_body", "filter_lower_guide")
    skid = _feature_bounds(features, "filter", "anti_roll_skid")
    clearances = channel_clearances(kerbs, skid, socket)
    assert min(clearances) >= 0.001 - FIT_TOLERANCE, f"Insufficient filter skid channel clearance: {clearances}"
    settled_z = support.GetMax()[2] - skid.GetMin()[2]
    engagement = []
    for kerb in kerbs:
        bounds = kerb.ComputeAlignedRange()
        assert bounds.GetSize()[1] >= 0.001 - FIT_TOLERANCE, "Filter kerb is too thin"
        for axis in (0, 1):
            assert bounds.GetMin()[axis] >= support.GetMin()[axis] - FIT_TOLERANCE, "Filter kerb overhangs support"
            assert bounds.GetMax()[axis] <= support.GetMax()[axis] + FIT_TOLERANCE, "Filter kerb overhangs support"
        assert abs(bounds.GetMin()[2] - support.GetMax()[2]) <= FIT_TOLERANCE, "Filter kerb must meet its support"
        overlap = bounds.GetMax()[2] - skid.GetMin()[2] - socket[2]
        assert overlap >= 0.0005 - FIT_TOLERANCE, "Filter kerb does not engage the nominal skid"
        engagement.append(overlap)
    for part in ("filter", "filter_decoy"):
        actual = _feature_bounds(features, part, "anti_roll_skid")
        declared = assets[part]["affordances"]["support_footprint_bounds"]
        for measured, expected in zip((actual.GetMin(), actual.GetMax()), declared):
            assert all(
                abs(measured[axis] - expected[axis]) <= FIT_TOLERANCE for axis in range(3)
            ), f"Filter support footprint differs from collision geometry: {part}"
    origin_bounds = [
        [socket[0], socket[1] - clearances[0], settled_z],
        [socket[0] + 0.075, socket[1] + clearances[1], socket[2]],
    ]
    non_skid = []
    for feature, shapes in features["filter"].items():
        if feature != "anti_roll_skid":
            non_skid.extend(shapes)
    report = {
        "capture": filter_capture_fit(support, kerbs, skid, socket, body["filter_capture_bounds"]),
        "side_clearances": list(clearances),
        "nominal_vertical_engagement": engagement,
        "settled_origin_z": settled_z,
        "extraction_origin_bounds": origin_bounds,
        "filter_running_clearance": translation_volume_clearance(non_skid, kerbs, origin_bounds, 0.0005),
        "skid_running_clearance": translation_volume_clearance(
            features["filter"]["anti_roll_skid"], kerbs, origin_bounds
        ),
    }
    endpoint = [socket[0] + 0.075, socket[1], settled_z]
    report["extracted_endpoint_clearance"] = translation_volume_clearance(
        non_skid + features["filter"]["anti_roll_skid"], kerbs, [endpoint, endpoint], 0.010
    )
    cup = body["cup_socket"]
    cup_shapes = []
    for shapes in features["dust_cup"].values():
        cup_shapes.extend(shapes)
    report["cup_prelift_and_extraction_clearance"] = translation_volume_clearance(
        cup_shapes, kerbs, [cup, [cup[0] + 0.10, cup[1], cup[2] + 0.003]], 0.005
    )
    return report


def validate_assembly_fits(assets: dict, features: dict) -> dict:
    """Check this task's guide fits, level supports and loose-part packing margins.

    Args:
        assets: Exported manifest entries, including sockets and placement poses.
        features: Actual exported collision bounds indexed by asset and feature.

    Returns:
        Measured clearances and support gaps in meters for the validation report.
    """
    report = {"channels": {}, "support_gaps": {}, "level_supports": {}, "packing": {}}
    channels = (
        ("battery_tester", "load_rail", "battery", "housing", "battery_socket", 0.004),
        ("vacuum_body", "battery_guide", "battery", "housing", "battery_socket", 0.003),
        ("vacuum_body", "filter_side_guide", "filter", "end_cap", "filter_socket", 0.003),
    )
    for fixture, guide, part, envelope, socket_name, minimum in channels:
        socket = assets[fixture]["affordances"][socket_name]
        rails = features[fixture].get(guide, [])
        bounds = _feature_bounds(features, part, envelope)
        clearances = channel_clearances(rails, bounds, socket)
        key = f"{fixture}/{guide}"
        assert min(clearances) >= minimum - FIT_TOLERANCE, f"Insufficient assembly clearance: {key}: {clearances}"
        report["channels"][key] = {"side_clearances": list(clearances), "minimum_required": minimum}

    decoy = _feature_bounds(features, "filter_decoy", "end_cap")
    decoy_clearances = channel_clearances(
        features["vacuum_body"]["filter_side_guide"], decoy, assets["vacuum_body"]["affordances"]["filter_socket"]
    )
    assert max(decoy_clearances) < -FIT_TOLERANCE, "Model B filter must remain too wide for the model A guides"
    report["decoy_filter_side_clearances"] = list(decoy_clearances)

    # Channel width alone misses corners interfering with the surrounding cup.
    # Oriented boxes preserve the hollow ring geometry that an AABB would fill.
    socket = assets["vacuum_body"]["affordances"]["cup_socket"]
    report["cup_internal_running_clearances"] = {}
    internal_parts = []
    for feature in ("filter_lower_guide", "filter_side_guide", "filter_skid_kerb"):
        internal_parts.extend(features["vacuum_body"][feature])
    filter_socket = assets["vacuum_body"]["affordances"]["filter_socket"]
    filter_transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*filter_socket))
    for shapes in features["filter"].values():
        for shape in shapes:
            internal_parts.append(Gf.BBox3d(shape.GetRange(), shape.GetMatrix() * filter_transform))
    for wall in ("collection_wall", "rear_sealing_lip"):
        minimum = extraction_clearance(
            features["dust_cup"][wall],
            internal_parts,
            (0.10, 0, 0),
            [-value for value in socket],
            minimum_required=0.003,
        )
        report["cup_internal_running_clearances"][wall] = {"separation_lower_bound": minimum, "minimum_required": 0.003}
    report["cup_seating_contact"] = axial_seating_fit(
        features["vacuum_body"]["service_socket"], features["dust_cup"]["rear_sealing_lip"], socket
    )

    # Parallel bridge connectors determine tester placement without scene imports.
    cup_socket = assets["dust_cup"]["affordances"]["airflow_adapter_socket"]
    bridge = assets["airflow_adapter"]["affordances"]
    tester_port = assets["airflow_tester"]["affordances"]["airflow_port"]
    tester_in_cup = []
    for axis in range(3):
        tester_in_cup.append(
            cup_socket[axis] + bridge["tester_connector"][axis] - bridge["cup_connector"][axis] - tester_port[axis]
        )
    cup_shapes = []
    for shapes in features["dust_cup"].values():
        cup_shapes.extend(shapes)
    clearance = extraction_clearance(
        cup_shapes, features["airflow_tester"]["display_housing"], (0.10, 0, 0), tester_in_cup
    )
    report["cup_extraction_display_clearance"] = {
        "separation_lower_bound": clearance,
        "minimum_required": 0.005,
        "extraction_distance": 0.10,
        "tester_origin_in_cup": tester_in_cup,
    }

    landing_support = []
    for feature in ("adapter_landing_post", "adapter_landing_pad"):
        landing_support.extend(features["airflow_tester"][feature])
    report["cup_extraction_landing_support_clearance"] = {
        "separation_lower_bound": extraction_clearance(
            cup_shapes, landing_support, (0.10, 0, 0), tester_in_cup, minimum_required=0.002
        ),
        "minimum_required": 0.002,
    }
    tester_in_adapter = [tester_in_cup[axis] - cup_socket[axis] for axis in range(3)]
    adapter_shapes = []
    for shapes in features["airflow_adapter"].values():
        adapter_shapes.extend(shapes)
    report["adapter_nominal_landing_support_clearance"] = {
        "separation_lower_bound": extraction_clearance(
            adapter_shapes, landing_support, (0.045, 0, 0), tester_in_adapter, minimum_required=0.0008
        ),
        "minimum_required": 0.0008,
    }

    body_in_cradle = assets["cradle"]["affordances"]["body_socket"]
    cup_in_body = assets["vacuum_body"]["affordances"]["cup_socket"]
    cup_in_cradle = [body_in_cradle[axis] + cup_in_body[axis] for axis in range(3)]
    report["skid_supports"] = {
        "dust_cup": support_fit(
            _feature_bounds(features, "cradle", "cup_slide_rail"),
            _feature_bounds(features, "dust_cup", "belly_skid"),
            cup_in_cradle,
            expected_gap=0.0,
        ),
        "filter": support_fit(
            _feature_bounds(features, "vacuum_body", "filter_lower_guide"),
            _feature_bounds(features, "filter", "anti_roll_skid"),
            filter_socket,
            expected_gap=0.0005,
        ),
    }
    cradle_shapes = []
    for shapes in features["cradle"].values():
        cradle_shapes.extend(shapes)
    # Sliding contact is intended; the extraction endpoint must fully clear
    # the rail before a controller corrects the cup's orientation or lifts it.
    report["cup_cradle_clearances"] = {
        "continuous_extraction": extraction_clearance(
            cup_shapes, cradle_shapes, (0.10, 0, 0), [-value for value in cup_in_cradle], minimum_required=0.0
        ),
        "extracted_endpoint": extraction_clearance(
            cup_shapes,
            cradle_shapes,
            (0, 0, 0),
            [-cup_in_cradle[0] - 0.10, -cup_in_cradle[1], -cup_in_cradle[2]],
            minimum_required=0.006,
        ),
    }

    supports = (
        ("battery_tester", "instrument_body", "battery", "housing", "battery_socket", 0.001),
        ("vacuum_body", "battery_support_shelf", "battery", "housing", "battery_socket", 0.0),
        ("vacuum_body", "filter_lower_guide", "filter", "end_cap", "filter_socket", 0.0005),
    )
    for fixture, support, part, envelope, socket_name, expected_gap in supports:
        socket = assets[fixture]["affordances"][socket_name]
        floor = _feature_bounds(features, fixture, support).GetMax()[2]
        bottom = _feature_bounds(features, part, envelope).GetMin()[2] + socket[2]
        gap = bottom - floor
        assert abs(gap - expected_gap) <= FIT_TOLERANCE, f"Changed support height: {fixture}/{support}: {gap}"
        report["support_gaps"][f"{fixture}/{support}"] = gap

    level_supports = (
        ("crevice_tool", "neck_rest", "connector"),
        ("nozzle", "neck_rest", "connector"),
        ("brush_tool", "bristle_tuft", "connector"),
        ("dust_cup", "belly_skid", "rear_sealing_lip"),
        ("filter", "anti_roll_skid", "end_cap"),
        ("filter_decoy", "anti_roll_skid", "end_cap"),
    )
    for part, support, reference in level_supports:
        reference_z = _feature_bounds(features, part, reference).GetMin()[2]
        shapes = features[part].get(support, [])
        assert shapes, f"Missing collision support: {part}/{support}"
        for shape in shapes:
            gap = shape.ComputeAlignedRange().GetMin()[2] - reference_z
            assert abs(gap) <= FIT_TOLERANCE, f"Nonlevel support: {part}/{support}: {gap}"
        report["level_supports"][part] = {"bottom_z": reference_z, "support_shape_count": len(shapes)}

    case = assets["case_base"]["affordances"]
    interior = case["interior_bounds"]
    placed = {}
    for part in ("battery", "crevice_tool", "brush_tool"):
        pose = case["packing_poses"][part]
        lower = [assets[part]["bounds_min"][i] + pose[i] for i in range(3)]
        upper = [assets[part]["bounds_max"][i] + pose[i] for i in range(3)]
        margins = []
        for axis in range(3):
            margins.extend((lower[axis] - interior[0][axis], interior[1][axis] - upper[axis]))
        assert min(margins) >= -FIT_TOLERANCE, f"Part exceeds nominal case packing bounds: {part}"
        placed[part] = (lower, upper)
        report["packing"][part] = {"boundary_margins": margins}
    for left, right in (("battery", "crevice_tool"), ("crevice_tool", "brush_tool")):
        gap = placed[right][0][0] - placed[left][1][0]
        assert gap >= 0.008 - FIT_TOLERANCE, f"Insufficient neighboring packing clearance: {left}/{right}: {gap}"
        report["packing"][f"{left}/{right}"] = {"x_gap": gap, "minimum_required": 0.008}
    brush_margin = interior[1][0] - placed["brush_tool"][1][0]
    assert brush_margin >= 0.010 - FIT_TOLERANCE, f"Insufficient brush-to-case clearance: {brush_margin}"
    report["case_latch"] = _case_latch_fits(assets, features)
    report["tray_interiors"] = {}
    for name in ("service_bin", "battery_bin", "filter_bin", "waste_bin", "parking_tray", "spare_rack"):
        floor = _feature_bounds(features, name, "floor")
        report["tray_interiors"][name] = tray_interior_fit(
            floor, features[name]["wall_x"], features[name]["wall_y"], assets[name]["affordances"]["interior_bounds"]
        )
    report["filter_keyed_guide"] = _filter_guide_fits(assets, features)
    report["obstruction_inlet"] = obstruction_inlet_fit(assets, features)
    return report
