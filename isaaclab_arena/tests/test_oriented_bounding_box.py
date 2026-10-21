# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometry checks for oriented bounding boxes."""

import math
import torch

import pytest

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, OrientedBoundingBox


def test_rotated_corners_and_axis_projection():
    """An asymmetric 3D rotation gives corners and projections independent of quaternion helpers."""
    quaternion = tuple(component / math.sqrt(30) for component in (1, 2, 3, 4))
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (2.0, 1.0, 0.5), quaternion)
    local_corners = torch.tensor([
        [-2.0, -1.0, -0.5],
        [2.0, -1.0, -0.5],
        [2.0, 1.0, -0.5],
        [-2.0, 1.0, -0.5],
        [-2.0, -1.0, 0.5],
        [2.0, -1.0, 0.5],
        [2.0, 1.0, 0.5],
        [-2.0, 1.0, 0.5],
    ])
    rotation = torch.tensor([[2 / 15, -2 / 3, 11 / 15], [14 / 15, 1 / 3, 2 / 15], [-1 / 3, 2 / 3, 2 / 3]])
    expected = (local_corners @ rotation.T + torch.tensor([1.0, 2.0, 3.0])).unsqueeze(0)
    torch.testing.assert_close(box.get_corners_at(), expected)
    torch.testing.assert_close(
        box.get_corners_at(torch.tensor([2.0, 0.0, 0.0])), expected + torch.tensor([2.0, 0.0, 0.0])
    )

    lower, upper = box.get_bounds_along_axis((2.0, 2.0, 0.0))
    projected = expected @ torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])
    torch.testing.assert_close(lower, projected.amin(dim=1))
    torch.testing.assert_close(upper, projected.amax(dim=1))
    lower, upper = box.get_axis_aligned_bounds()
    torch.testing.assert_close(lower, expected.amin(dim=1))
    torch.testing.assert_close(upper, expected.amax(dim=1))
    enclosing = box.to_aabb()
    torch.testing.assert_close(enclosing.min_point, expected.amin(dim=1))
    torch.testing.assert_close(enclosing.max_point, expected.amax(dim=1))
    with pytest.raises(AssertionError):
        box.get_bounds_along_axis((0.0, 0.0, 0.0))


def test_transform_composes_rotations_and_rotates_the_center():
    """A quarter-turn about X followed by Z maps local (x, y, z) to (z, x, y)."""
    s = math.sqrt(0.5)
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (s, 0.0, 0.0, s))
    transformed = box.transformed((4.0, 5.0, 6.0), (0.0, 0.0, s, s))

    torch.testing.assert_close(transformed.center, torch.tensor([[2.0, 6.0, 9.0]]))
    torch.testing.assert_close(transformed.rotation_xyzw, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    torch.testing.assert_close(
        transformed.get_corners_at(),
        torch.tensor([[
            [-1.0, 5.0, 7.0],
            [-1.0, 7.0, 7.0],
            [-1.0, 7.0, 11.0],
            [-1.0, 5.0, 11.0],
            [5.0, 5.0, 7.0],
            [5.0, 7.0, 7.0],
            [5.0, 7.0, 11.0],
            [5.0, 5.0, 11.0],
        ]]),
    )
    torch.testing.assert_close(box.center, torch.tensor([[1.0, 2.0, 3.0]]))


def test_aabb_transform_matches_enclosing_rotation():
    """An exactly transformed AABB has the same axis-aligned bounds as rotated_by_quat."""
    box = AxisAlignedBoundingBox((-2.0, -1.0, -0.5), (2.0, 1.0, 0.5))
    quaternion = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    exact = box.transformed((0.0, 0.0, 0.0), quaternion)
    enclosing = box.rotated_by_quat(quaternion)

    assert isinstance(box, OrientedBoundingBox)
    assert type(exact) is OrientedBoundingBox
    assert type(enclosing) is AxisAlignedBoundingBox
    torch.testing.assert_close(exact.half_extents, torch.tensor([[2.0, 1.0, 0.5]]))
    lower, upper = exact.get_axis_aligned_bounds()
    torch.testing.assert_close(enclosing.min_point, lower)
    torch.testing.assert_close(enclosing.max_point, upper)
    torch.testing.assert_close(box.to_obb().get_corners_at(), box.get_corners_at())


def test_sat_separates_boxes_whose_six_face_axes_overlap():
    """Skew rods separate on an edge-cross-edge axis, despite all face projections overlapping."""
    first = OrientedBoundingBox((0.0, 0.0, 0.0), (2.0, 0.1, 0.1))
    second = OrientedBoundingBox(
        (0.0, 0.2612789059, 0.1847520861),
        (2.0, 0.1, 0.1),
        (-0.1464466094, 0.3535533906, 0.3535533906, 0.8535533906),
    )
    s = math.sqrt(0.5)
    face_axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, -s), (-s, s, 0.0), (0.5, 0.5, s)]
    for axis in face_axes:
        first_min, first_max = first.get_bounds_along_axis(axis)
        second_min, second_max = second.get_bounds_along_axis(axis)
        assert (first_max > second_min).all() and (second_max > first_min).all()

    assert not first.overlaps(second).item()
    assert not second.overlaps(first).item()
    assert not first.overlaps(second, margin=0.04).item()
    assert first.overlaps(second, margin=0.05).item()
    torch.testing.assert_close(first.penetration_depth(second), torch.zeros(1))

    separating_direction = torch.tensor([0.0, math.sqrt(2 / 3), 1 / math.sqrt(3)])
    overlapping = second.translated(-0.08 * separating_direction)
    expected_depth = 0.2 * (math.sqrt(2 / 3) + 1 / math.sqrt(3)) - 0.24
    torch.testing.assert_close(first.penetration_depth(overlapping), torch.tensor([expected_depth]))
    torch.testing.assert_close(overlapping.penetration_depth(first), torch.tensor([expected_depth]))
    assert first.overlaps(overlapping.translated((expected_depth - 1e-4) * separating_direction)).item()
    assert not first.overlaps(overlapping.translated((expected_depth + 1e-4) * separating_direction)).item()


def test_nearly_parallel_thin_boxes_do_not_use_their_enclosing_aabbs():
    """Rods with distinct tiny yaws separate despite their enclosing AABBs overlapping."""
    angle = 5e-6
    first = OrientedBoundingBox(
        (0.0, 0.0, 0.0), (100.0, 0.001, 1.0), (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    )
    second_angle = 6e-6
    second = OrientedBoundingBox(
        (0.0, 0.0023, 0.0),
        (100.0, 0.001, 1.0),
        (0.0, 0.0, math.sin(second_angle / 2), math.cos(second_angle / 2)),
    )
    first_bounds = AxisAlignedBoundingBox(*first.get_axis_aligned_bounds())
    second_bounds = AxisAlignedBoundingBox(*second.get_axis_aligned_bounds())

    assert first_bounds.overlaps(second_bounds).item()
    assert not first.overlaps(second).item()
    assert not second.overlaps(first).item()
    assert first._sat_depths(second, margin=0.0)[0].shape == (1, 15)


def test_mixed_box_overlap_is_symmetric_with_strict_margins():
    """AABB callers use exact OBB geometry, and touching at the requested margin is allowed."""
    first = AxisAlignedBoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    touching = OrientedBoundingBox((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    separated = touching.translated((0.125, 0.0, 0.0))
    with pytest.raises(AssertionError):
        touching.overlaps(first, margin=-0.125)
    for left, right in ((first, touching), (touching, first)):
        assert not left.overlaps(right).item()
        assert left.overlaps(right, margin=0.125).item()
    torch.testing.assert_close(touching.penetration_depth(first, margin=0.125), torch.tensor([0.125]))
    torch.testing.assert_close(first.penetration_depth(touching, margin=0.125), torch.tensor([0.125]))
    for left, right in ((first, separated), (separated, first)):
        assert not left.overlaps(right, margin=0.125).item()
        assert left.overlaps(right, margin=0.25).item()

    point = OrientedBoundingBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert first.overlaps(point).item()
    assert not first.overlaps(point.translated((1.0, 0.0, 0.0))).item()

    small = AxisAlignedBoundingBox((-0.1, 0.3, -0.05), (0.1, 0.5, 0.05))
    rod = OrientedBoundingBox(
        (0.0, 0.0, 0.0), (1.0, 0.05, 0.05), (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    )
    assert not small.overlaps(rod).item()
    assert not rod.overlaps(small).item()
    assert small.translated((0.0, -0.4, 0.0)).overlaps(rod).item()


def test_box_batches_keep_rows_through_projection_and_transforms():
    """Singleton extents broadcast without mixing different environments' centers or rotations."""
    s = math.sqrt(0.5)
    boxes = OrientedBoundingBox(
        torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
        (1.0, 2.0, 0.5),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, s, s]]),
    )
    assert boxes.num_envs == 2 and boxes.center.dtype == torch.float32
    lower, upper = boxes.get_bounds_along_axis((1.0, 0.0, 0.0))
    torch.testing.assert_close(lower, torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(upper, torch.tensor([1.0, 6.0]))
    target = OrientedBoundingBox((2.5, 0.0, 0.0), (0.1, 0.1, 0.1))
    assert boxes.overlaps(target).tolist() == [False, True]
    assert target.overlaps(boxes).tolist() == [False, True]
    moved = boxes.transformed(
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]),
        torch.tensor([[0.0, 0.0, s, s], [s, 0.0, 0.0, s]]),
    )
    expected_centers = torch.tensor([[1.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    expected_rotations = torch.tensor([[0.0, 0.0, s, s], [0.5, -0.5, 0.5, 0.5]])
    torch.testing.assert_close(moved.center, expected_centers)
    torch.testing.assert_close(moved.rotation_xyzw, expected_rotations)
    local_corners = torch.tensor([
        [-1.0, -2.0, -0.5],
        [1.0, -2.0, -0.5],
        [1.0, 2.0, -0.5],
        [-1.0, 2.0, -0.5],
        [-1.0, -2.0, 0.5],
        [1.0, -2.0, 0.5],
        [1.0, 2.0, 0.5],
        [-1.0, 2.0, 0.5],
    ])
    first_corners = torch.stack((-local_corners[:, 1], local_corners[:, 0], local_corners[:, 2]), dim=-1)
    second_corners = torch.stack((-local_corners[:, 1], -local_corners[:, 2], local_corners[:, 0]), dim=-1)
    expected_corners = torch.stack((first_corners, second_corners)) + expected_centers[:, None, :]
    torch.testing.assert_close(moved.get_corners_at(), expected_corners)
    assert moved[1].num_envs == 1
    torch.testing.assert_close(moved[1].get_corners_at(), expected_corners[1:2])
    torch.testing.assert_close(moved[1].to("cpu").rotation_xyzw, expected_rotations[1:2])
    with pytest.raises(AssertionError):
        OrientedBoundingBox(torch.zeros(2, 3), torch.ones(3, 3))


def test_penetration_is_separation_depth_with_translation_gradients():
    """Penetration measures escape distance, including containment, and differentiates by translation."""
    first = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    center = torch.tensor([[1.5, 0.1, 0.2]], requires_grad=True)
    extents = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
    s = math.sqrt(0.5)
    second = OrientedBoundingBox(center, extents, (0.0, 0.0, s, s))
    depth = first.penetration_depth(second)
    torch.testing.assert_close(depth, torch.tensor([0.5]))
    depth.sum().backward()
    torch.testing.assert_close(center.grad, torch.tensor([[-1.0, 0.0, 0.0]]))
    torch.testing.assert_close(extents.grad, torch.tensor([[0.0, 1.0, 0.0]]))

    outer = OrientedBoundingBox((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    inner = OrientedBoundingBox((0.5, 0.0, 0.0), (1.0, 1.0, 1.0))
    torch.testing.assert_close(outer.penetration_depth(inner), torch.tensor([3.5]))


def test_aabb_empty_and_unbounded_regions_preserve_interval_operations():
    """Empty and infinite intervals keep their exact ranges but reject finite-geometry queries."""
    bounded = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    empty = bounded.intersected(AxisAlignedBoundingBox((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)))
    unbounded = AxisAlignedBoundingBox((-math.inf, -math.inf, -math.inf), (math.inf, math.inf, math.inf))
    assert empty.is_empty.tolist() == [True]
    assert not bounded.overlaps(empty).item()
    assert bounded.overlaps(unbounded).item()
    torch.testing.assert_close(bounded.intersected(unbounded).max_point, bounded.max_point)
    oriented = bounded.to_obb()
    for box in (empty, unbounded):
        expected_lower, expected_upper = box.min_point.clone(), box.max_point.clone()
        lower, upper = box.get_axis_aligned_bounds()
        copied = box.to_aabb()
        for actual in (lower, copied.min_point):
            torch.testing.assert_close(actual, expected_lower, rtol=0, atol=0)
            actual.zero_()
        for actual in (upper, copied.max_point):
            torch.testing.assert_close(actual, expected_upper, rtol=0, atol=0)
            actual.zero_()
        torch.testing.assert_close(box.min_point, expected_lower, rtol=0, atol=0)
        torch.testing.assert_close(box.max_point, expected_upper, rtol=0, atol=0)
        with pytest.raises(AssertionError):
            box.to_obb()
        with pytest.raises(AssertionError):
            box.transformed((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        with pytest.raises(AssertionError):
            box.get_bounds_along_axis((1.0, 0.0, 0.0))
        with pytest.raises(AssertionError):
            box.penetration_depth(bounded)
        with pytest.raises(AssertionError):
            bounded.penetration_depth(box)
        with pytest.raises(AssertionError):
            box.overlaps(oriented)
        with pytest.raises(AssertionError):
            oriented.overlaps(box)


def test_aabb_inherited_queries_follow_edits_to_bounds():
    """Inherited transforms, projections and penetration read current min/max tensor views."""
    box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))
    other = AxisAlignedBoundingBox((1.5, 0.0, 0.0), (3.5, 4.0, 6.0))
    torch.testing.assert_close(box.penetration_depth(other), torch.tensor([0.5]))
    box[0].min_point[0, 0] = 0.5
    box.max_point[0, :2] = torch.tensor([3.0, 6.0])

    lower, upper = box.get_bounds_along_axis((1.0, 2.0, 2.0))
    torch.testing.assert_close(lower, torch.tensor([1 / 6]))
    torch.testing.assert_close(upper, torch.tensor([9.0]))
    torch.testing.assert_close(box.penetration_depth(other), torch.tensor([1.5]))
    s = math.sqrt(0.5)
    moved = box.transformed((4.0, 5.0, 6.0), (0.0, 0.0, s, s))
    torch.testing.assert_close(moved.center, torch.tensor([[1.0, 6.75, 9.0]]))
    lower, upper = moved.get_axis_aligned_bounds()
    torch.testing.assert_close(lower, torch.tensor([[-2.0, 5.5, 6.0]]))
    torch.testing.assert_close(upper, torch.tensor([[4.0, 8.0, 12.0]]))


def test_aabb_conversion_copies_current_bounds():
    """A conversion captures current AABB bounds without sharing their mutable storage."""
    box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    original = box.to_obb()
    box[0].min_point[0, 0] = 0.5
    box.max_point[0, 1] = 4.0
    converted = box.to_obb()
    torch.testing.assert_close(converted.center, torch.tensor([[1.25, 2.0, 1.0]]))
    torch.testing.assert_close(converted.half_extents, torch.tensor([[0.75, 2.0, 1.0]]))
    torch.testing.assert_close(converted.get_corners_at(), box.get_corners_at())
    torch.testing.assert_close(original.center, torch.ones(1, 3))
    torch.testing.assert_close(original.half_extents, torch.ones(1, 3))


def test_constructor_rejects_invalid_geometry():
    """Non-finite bounds, negative extents and non-unit quaternions cannot describe an OBB."""
    invalid = [
        ((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (math.inf, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 2.0)),
    ]
    for center, half_extents, rotation in invalid:
        with pytest.raises(AssertionError):
            OrientedBoundingBox(center, half_extents, rotation)


def test_input_and_property_writes_do_not_change_box_geometry():
    """Input tensors and broadcast property rows can be edited without corrupting a box."""
    centers = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    extents = torch.ones(1, 3)
    rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    box = OrientedBoundingBox(centers, extents, rotation)
    corners = box.get_corners_at()
    centers.add_(10)
    extents.zero_()
    rotation.zero_()
    torch.testing.assert_close(box.get_corners_at(), corners)

    returned_extents = box.half_extents
    returned_extents[0, 0] = 5
    assert returned_extents[1, 0] == 1
    returned_extents.mul_(2)
    box.center.add_(10)
    box.rotation_xyzw.zero_()
    torch.testing.assert_close(box.get_corners_at(), corners)


def test_rounded_quaternions_are_normalized_for_construction_and_transforms():
    """Rounded authored quarter-turns have the same geometry as unit quaternions."""
    rounded = (0.0, 0.0, 0.707, 0.707)
    exact = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    reference = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 0.5), exact)
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (1.0, 2.0, 0.5), rounded)
    torch.testing.assert_close(box.rotation_xyzw, reference.rotation_xyzw)
    torch.testing.assert_close(box.get_corners_at(), reference.get_corners_at())
    torch.testing.assert_close(
        box.transformed((0.0, 0.0, 0.0), rounded).get_corners_at(),
        reference.transformed((0.0, 0.0, 0.0), exact).get_corners_at(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_box_queries_preserve_geometry_and_device_on_cuda():
    """Device transfers preserve batched rotations and queries reject mixed-device inputs."""
    boxes = OrientedBoundingBox(
        torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]]),
        (1.0, 2.0, 0.5),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]),
    )
    target = OrientedBoundingBox((0.75, 0.0, 0.0), (0.1, 0.1, 0.1))
    gpu_boxes, gpu_target = boxes.to("cuda"), target.to("cuda")
    torch.testing.assert_close(gpu_boxes.get_corners_at().cpu(), boxes.get_corners_at())
    torch.testing.assert_close(gpu_boxes.to_aabb().max_point.cpu(), boxes.to_aabb().max_point)
    torch.testing.assert_close(gpu_boxes.overlaps(gpu_target).cpu(), boxes.overlaps(target))
    torch.testing.assert_close(gpu_boxes.penetration_depth(gpu_target).cpu(), boxes.penetration_depth(target))
    with pytest.raises(AssertionError, match="device"):
        gpu_boxes.overlaps(target)

    gpu_aabb = AxisAlignedBoundingBox((0.1, 0.2, 0.3), (1.0, 2.0, 3.0)).to(torch.device("cuda"))
    shifted = gpu_aabb.translated((1.0, 2.0, 3.0))
    assert shifted.min_point.device == gpu_aabb.min_point.device
    offset = torch.tensor([[1.0, 2.0, 3.0]], device=gpu_aabb.min_point.device)
    torch.testing.assert_close(shifted.min_point, gpu_aabb.min_point + offset)
    torch.testing.assert_close(shifted.max_point, gpu_aabb.max_point + offset)
    projected_lower, projected_upper = shifted.get_bounds_along_axis((1.0, 0.0, 0.0))
    torch.testing.assert_close(projected_lower, shifted.min_point[:, 0])
    torch.testing.assert_close(projected_upper, shifted.max_point[:, 0])


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_axis_permutations_use_three_axes_without_generic_sat(device, monkeypatch):
    """Exact axis permutations preserve batched depth, containment and margins without fifteen-axis work."""
    s = math.sqrt(0.5)
    boxes = OrientedBoundingBox(
        torch.tensor([[1.125, 0, 0], [0.375, 0, 0], [0.625, 0, 0], [0.75, 0, 0]], device=device),
        (1.0, 0.25, 0.5),
        torch.tensor([[0, 0, 0, 1], [0, 0, -s, -s], [0.5, 0.5, 0.5, 0.5], [s, s, 0, 0]], device=device),
    )
    target = OrientedBoundingBox(torch.zeros(3, device=device), (0.25, 0.25, 0.25))
    containing = OrientedBoundingBox(torch.zeros(3, device=device), (3.0, 3.0, 3.0))
    tuple_rotation = OrientedBoundingBox(torch.tensor([0.375, 0, 0], device=device), (1.0, 0.25, 0.5), (0, 0, -s, -s))

    def unexpected_einsum(*args, **kwargs):
        pytest.fail("Axis-aligned box queries must not compute generic SAT projections")

    monkeypatch.setattr(torch, "einsum", unexpected_einsum)
    assert boxes._sat_depths(target, margin=0.0)[0].shape == (4, 3)
    assert boxes.overlaps(target).tolist() == [True, True, True, False]
    assert target.overlaps(boxes).tolist() == [True, True, True, False]
    expected = torch.tensor([0.125, 0.125, 0.125, 0.0], device=device)
    torch.testing.assert_close(boxes.penetration_depth(target), expected)
    torch.testing.assert_close(target.penetration_depth(boxes), expected)
    assert boxes.overlaps(target, margin=0.25).tolist() == [True, True, True, False]
    assert boxes.overlaps(target, margin=0.5).all()
    torch.testing.assert_close(
        boxes.penetration_depth(target, margin=0.5), torch.tensor([0.625, 0.625, 0.625, 0.25], device=device)
    )
    torch.testing.assert_close(
        boxes.penetration_depth(containing), torch.tensor([2.875, 2.875, 2.875, 2.5], device=device)
    )
    torch.testing.assert_close(tuple_rotation.penetration_depth(target), torch.tensor([0.125], device=device))


def test_near_quarter_turn_and_mixed_rows_keep_full_sat():
    """An 89.9-degree box overlaps where snapping to a quarter-turn would report touching."""
    angle = math.radians(89.9)
    boxes = OrientedBoundingBox(
        torch.zeros(2, 3),
        (1.0, 0.25, 0.5),
        torch.tensor([[0, 0, 0, 1], [0, 0, math.sin(angle / 2), math.cos(angle / 2)]]),
    )
    target = OrientedBoundingBox((0.3, 0.95, 0.0), (0.05, 0.05, 0.05))
    assert boxes._sat_depths(target, margin=0.0)[0].shape == (2, 15)
    assert boxes[1]._sat_depths(target, margin=0.0)[0].shape == (1, 15)
    assert boxes.overlaps(target).tolist() == [False, True]
    expected_depth = math.cos(angle) + 0.25 * math.sin(angle) - 0.25
    torch.testing.assert_close(
        boxes.penetration_depth(target), torch.tensor([0.0, expected_depth]), atol=1e-6, rtol=1e-5
    )


def test_axis_aligned_rotation_gradients_use_generic_sat():
    """A trainable quarter-turn retains the orientation gradient from the full SAT calculation."""
    s = math.sqrt(0.5)
    rotation = torch.tensor([0.0, 0.0, s, s], requires_grad=True)
    box = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.25, 0.5), rotation)
    target = OrientedBoundingBox((0.375, 0.1, 0.0), (0.25, 0.25, 0.25))
    reference = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.25, 0.5), rotation)
    reference._axis_order = None
    assert box._sat_depths(target, margin=0.0)[0].shape == (1, 15)
    depth = box.penetration_depth(target)
    expected_depth = reference.penetration_depth(target)
    gradient = torch.autograd.grad(depth.sum(), rotation)[0]
    expected_gradient = torch.autograd.grad(expected_depth.sum(), rotation)[0]
    torch.testing.assert_close(depth, expected_depth)
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.count_nonzero(gradient) > 0
