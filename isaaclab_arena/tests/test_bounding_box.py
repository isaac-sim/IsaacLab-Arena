# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for oriented bounding-box geometry."""

import math
import torch

import pytest

from isaaclab_arena.utils.bounding_box import OrientedBoundingBox, get_random_pose_within_bounding_box

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _yaw(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def _pitch(angle: float) -> tuple[float, float, float, float]:
    return (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))


def test_construction_and_min_max():
    """Construction and min/max conversion preserve box geometry."""
    box = OrientedBoundingBox((1, 2, 3), (0.5, 1, 2), IDENTITY)
    assert box.num_envs == 1
    assert box.center.shape == box.half_extents.shape == (1, 3)
    assert box.rotation_xyzw.shape == (1, 4)
    assert box.center.dtype == box.half_extents.dtype == box.rotation_xyzw.dtype == torch.float32

    from_bounds = OrientedBoundingBox.from_min_max((-1, 0, 2), (3, 4, 8))
    torch.testing.assert_close(from_bounds.center, torch.tensor([[1.0, 2.0, 5.0]]))
    torch.testing.assert_close(from_bounds.half_extents, torch.tensor([[2.0, 2.0, 3.0]]))
    torch.testing.assert_close(from_bounds.rotation_xyzw, torch.tensor([IDENTITY]))


def test_constructor_clones_tensor_inputs():
    """Construction clones mutable tensor inputs."""
    center = torch.tensor([1.0, 2.0, 3.0])
    half_extents = torch.tensor([0.5, 1.0, 1.5])
    rotation = torch.tensor(IDENTITY)
    box = OrientedBoundingBox(center, half_extents, rotation)

    center.fill_(float("nan"))
    half_extents.fill_(-1.0)
    rotation.fill_(2.0)

    torch.testing.assert_close(box.center, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(box.half_extents, torch.tensor([[0.5, 1.0, 1.5]]))
    torch.testing.assert_close(box.rotation_xyzw, torch.tensor([IDENTITY]))


def test_batch_broadcast_index_invariance_and_translation():
    """Batched boxes broadcast, index, classify, and translate correctly."""
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    box = OrientedBoundingBox(centers, (0.5, 1.0, 1.5), IDENTITY)
    assert box.num_envs == 2
    assert box.half_extents.shape == box.rotation_xyzw.shape[:1] + (3,)
    assert not box.is_batch_invariant()
    torch.testing.assert_close(box[1].center, centers[1:])
    with pytest.raises(AssertionError):
        _ = box[2]

    moved = box.translated(torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0]]))
    torch.testing.assert_close(moved.center, torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 5.0]]))
    invariant = OrientedBoundingBox(torch.zeros(2, 3), torch.ones(2, 3), torch.tensor([IDENTITY]))
    assert invariant.is_batch_invariant()
    assert invariant.to("cpu").center.device.type == "cpu"


def test_off_origin_rotation_and_transform_broadcast():
    """Rigid transforms preserve xyzw composition and row broadcasting."""
    box = OrientedBoundingBox((1.0, 0.0, 0.0), (0.5, 0.25, 0.1), IDENTITY)
    rotated = box.rotated_by_quat(_yaw(math.pi / 2))
    torch.testing.assert_close(rotated.center, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated.rotation_xyzw, torch.tensor([_yaw(math.pi / 2)]), atol=1e-6, rtol=0)

    transformed = box.transformed(
        torch.tensor([[10.0, 0.0, 1.0], [20.0, 0.0, 2.0]]),
        torch.tensor([_yaw(math.pi / 2), _yaw(math.pi)]),
    )
    assert transformed.num_envs == 2
    torch.testing.assert_close(
        transformed.center,
        torch.tensor([[10.0, 1.0, 1.0], [19.0, 0.0, 2.0]]),
        atol=1e-6,
        rtol=0,
    )


def test_corners_and_axis_bounds():
    """Corners and projected bounds agree for a yawed box."""
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (2.0, 1.0, 0.5), _yaw(math.pi / 2))
    corners = box.get_corners()
    assert corners.shape == (1, 8, 3)
    torch.testing.assert_close(corners.mean(dim=1), box.center, atol=1e-6, rtol=0)
    expected_min = torch.tensor([[0.0, 0.0, 2.5]])
    expected_max = torch.tensor([[2.0, 4.0, 3.5]])
    minimum, maximum = box.get_axis_aligned_bounds()
    torch.testing.assert_close(minimum, expected_min, atol=1e-6, rtol=0)
    torch.testing.assert_close(maximum, expected_max, atol=1e-6, rtol=0)
    torch.testing.assert_close(corners.amin(dim=1), expected_min, atol=1e-6, rtol=0)
    torch.testing.assert_close(corners.amax(dim=1), expected_max, atol=1e-6, rtol=0)

    lower, upper = box.get_bounds_along_axis((2.0, 0.0, 0.0))
    torch.testing.assert_close(lower, torch.tensor([0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(upper, torch.tensor([2.0]), atol=1e-6, rtol=0)


def test_batched_corners_use_each_rotation():
    """Batched corners apply each row's distinct xyzw rotation."""
    box = OrientedBoundingBox(
        torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        (2.0, 1.0, 0.5),
        torch.tensor([_yaw(math.pi / 2), _pitch(math.pi / 2)]),
    )

    corners = box.get_corners()

    torch.testing.assert_close(corners[0].amin(dim=0), torch.tensor([-1.0, -2.0, -0.5]), atol=1e-6, rtol=0)
    torch.testing.assert_close(corners[0].amax(dim=0), torch.tensor([1.0, 2.0, 0.5]), atol=1e-6, rtol=0)
    torch.testing.assert_close(corners[1].amin(dim=0), torch.tensor([9.5, -1.0, -2.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(corners[1].amax(dim=0), torch.tensor([10.5, 1.0, 2.0]), atol=1e-6, rtol=0)


def test_axis_bounds_broadcast_axes():
    """Projection axes broadcast over a single box."""
    box = OrientedBoundingBox((1.0, 2.0, 3.0), (0.5, 1.0, 2.0), IDENTITY)
    lower, upper = box.get_bounds_along_axis(torch.eye(3))
    torch.testing.assert_close(lower, torch.tensor([0.5, 1.0, 1.0]))
    torch.testing.assert_close(upper, torch.tensor([1.5, 3.0, 5.0]))


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (IDENTITY, True),
        (_yaw(math.pi / 2), True),
        (_yaw(math.pi), True),
        (_yaw(-math.pi / 2), True),
        (_pitch(math.pi / 2), True),
        (_yaw(math.pi / 4), False),
    ],
)
def test_axis_alignment_signed_permutations(rotation, expected):
    """Signed axis permutations are classified as axis aligned."""
    box = OrientedBoundingBox((0, 0, 0), (1, 2, 3), rotation)
    assert box.is_axis_aligned().item() is expected


def test_aabb_overlap_fast_path_touching_and_clearance():
    """The axis-aligned path handles overlap, contact, and clearance."""
    box = OrientedBoundingBox.from_min_max((0, 0, 0), (1, 1, 1))
    overlapping = OrientedBoundingBox.from_min_max((0.75, 0, 0), (1.75, 1, 1))
    touching = OrientedBoundingBox.from_min_max((1, 0, 0), (2, 1, 1))
    separated = OrientedBoundingBox.from_min_max((1.1, 0, 0), (2.1, 1, 1))
    assert box.overlaps(overlapping).item() is True
    torch.testing.assert_close(box.penetration(overlapping), torch.tensor([0.25]))
    assert box.overlaps(touching).item() is False
    torch.testing.assert_close(box.penetration(touching), torch.tensor([0.0]))
    assert box.overlaps(separated).item() is False
    assert box.overlaps(separated, clearance_m=0.2).item() is True
    torch.testing.assert_close(box.penetration(separated, clearance_m=0.2), torch.tensor([0.1]), atol=1e-6, rtol=0)


def test_rotated_sat_separation_penetration_and_clearance():
    """The SAT path handles separation, penetration, and clearance."""
    rotation = _yaw(math.pi / 4)
    box = OrientedBoundingBox((0, 0, 0), (1.0, 0.2, 0.2), rotation)
    perpendicular = torch.tensor([-math.sqrt(0.5), math.sqrt(0.5), 0.0])
    separated = OrientedBoundingBox(perpendicular * 0.5, (1.0, 0.2, 0.2), rotation)
    close = OrientedBoundingBox(perpendicular * 0.3, (1.0, 0.2, 0.2), rotation)
    touching = OrientedBoundingBox(perpendicular * 0.4, (1.0, 0.2, 0.2), rotation)

    min_a, max_a = box.get_axis_aligned_bounds()
    min_b, max_b = separated.get_axis_aligned_bounds()
    assert ((max_a > min_b) & (max_b > min_a)).all()
    assert box.overlaps(separated).item() is False
    assert box.overlaps(close).item() is True
    torch.testing.assert_close(box.penetration(close), torch.tensor([0.1]), atol=1e-5, rtol=0)
    assert box.overlaps(touching).item() is False
    assert box.overlaps(separated, clearance_m=0.2).item() is True
    torch.testing.assert_close(box.penetration(separated, clearance_m=0.2), torch.tensor([0.1]), atol=1e-5, rtol=0)


def test_identity_fast_path_matches_tiny_angle_sat_path():
    """Identity fast-path depths match tiny-angle SAT depths in mixed batches."""
    centers = torch.tensor([[0.25, 0.1, 0.0], [0.25, 0.1, 0.0]])
    identity_boxes = OrientedBoundingBox(centers, (1.0, 0.8, 0.6), IDENTITY)
    mixed_boxes = OrientedBoundingBox(centers, (1.0, 0.8, 0.6), torch.tensor([IDENTITY, _yaw(1e-4)]))
    obstacle = OrientedBoundingBox((0.0, 0.0, 0.0), (0.7, 0.5, 0.4), IDENTITY)

    fast = identity_boxes.penetration(obstacle)
    mixed = mixed_boxes.penetration(obstacle)

    torch.testing.assert_close(mixed[0], fast[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(mixed[1], fast[1], atol=1e-4, rtol=0)


def test_overlap_batch_broadcast_and_gradients():
    """Penetration produces finite nonzero gradients in overlapping rows."""
    centers = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], requires_grad=True)
    boxes = OrientedBoundingBox(centers, (1, 1, 1), _yaw(math.pi / 4))
    other = OrientedBoundingBox((0.5, 0, 0), (1, 1, 1), IDENTITY)
    result = boxes.overlaps(other)
    assert result.tolist() == [True, False]
    loss = boxes.penetration(other).sum()
    loss.backward()
    assert centers.grad is not None
    assert torch.isfinite(centers.grad).all()
    assert torch.count_nonzero(centers.grad[0]).item() > 0


@pytest.mark.parametrize("rotation", [IDENTITY, _yaw(math.pi / 4)])
def test_concentric_penetration_has_nonzero_center_gradient(rotation):
    """Concentric axis-aligned and rotated boxes retain a deterministic escape gradient."""
    center = torch.zeros(3, requires_grad=True)
    box = OrientedBoundingBox(center, (1.0, 0.6, 0.4), rotation)
    obstacle = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.6, 0.4), rotation)

    penetration = box.penetration(obstacle)
    penetration.sum().backward()

    torch.testing.assert_close(penetration, torch.tensor([0.8]), atol=1e-6, rtol=0)
    assert center.grad is not None
    assert torch.isfinite(center.grad).all()
    assert torch.count_nonzero(center.grad).item() > 0


def test_concentric_cross_axis_tie_breaks_produce_opposite_gradients():
    """Directed ties use one canonical SAT cross axis when box order reverses."""
    from isaaclab.utils.math import quat_from_euler_xyz

    rotation_a = quat_from_euler_xyz(
        torch.tensor(0.54799175),
        torch.tensor(0.74369037),
        torch.tensor(-0.25836289),
    )
    rotation_b = quat_from_euler_xyz(
        torch.tensor(0.23610878),
        torch.tensor(-0.42937827),
        torch.tensor(-0.71320212),
    )
    centers = [torch.zeros(3, requires_grad=True) for _ in range(2)]
    boxes = [
        OrientedBoundingBox(centers[0], (2.0, 0.4, 0.15), rotation_a),
        OrientedBoundingBox(centers[1], (1.7, 0.3, 0.2), rotation_b),
    ]

    forward = boxes[0].penetration(
        OrientedBoundingBox(boxes[1].center.detach(), boxes[1].half_extents, boxes[1].rotation_xyzw),
        tie_break_sign=1.0,
    )
    reverse = boxes[1].penetration(
        OrientedBoundingBox(boxes[0].center.detach(), boxes[0].half_extents, boxes[0].rotation_xyzw),
        tie_break_sign=-1.0,
    )
    (forward + reverse).sum().backward()

    assert centers[0].grad is not None
    assert centers[1].grad is not None
    assert torch.count_nonzero(centers[0].grad).item() > 0
    torch.testing.assert_close(centers[0].grad, -centers[1].grad, atol=1e-6, rtol=0)


def test_penetration_tie_break_does_not_change_nonconcentric_result():
    """Directed tie signs leave non-tied penetration values and gradients unchanged."""
    centers = [torch.tensor([0.3, -0.1, 0.1], requires_grad=True) for _ in range(2)]
    obstacle = OrientedBoundingBox((0.0, 0.0, 0.0), (1.0, 0.6, 0.4), _yaw(math.pi / 4))
    values = []
    gradients = []

    for center, tie_break_sign in zip(centers, (1.0, -1.0)):
        value = OrientedBoundingBox(center, (1.0, 0.6, 0.4), _yaw(math.pi / 4)).penetration(
            obstacle, tie_break_sign=tie_break_sign
        )
        value.sum().backward()
        values.append(value)
        gradients.append(center.grad)

    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: OrientedBoundingBox((0, 0), (1, 1, 1), IDENTITY),
        lambda: OrientedBoundingBox((0, 0, 0), (-1, 1, 1), IDENTITY),
        lambda: OrientedBoundingBox((0, 0, 0), (1, 1, 1), (0, 0, 0, 2)),
        lambda: OrientedBoundingBox(torch.zeros(2, 3), torch.ones(3, 3), IDENTITY),
        lambda: OrientedBoundingBox.from_min_max((1, 0, 0), (0, 1, 1)),
        lambda: OrientedBoundingBox(torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0, 4))),
        lambda: OrientedBoundingBox((float("nan"), 0, 0), (1, 1, 1), IDENTITY),
        lambda: OrientedBoundingBox((0, 0, 0), (float("inf"), 1, 1), IDENTITY),
    ],
)
def test_invalid_invariants(constructor):
    """Invalid dimensions, batches, ranges, and values are rejected."""
    with pytest.raises(AssertionError):
        constructor()


def test_invalid_clearance_is_rejected():
    """Negative SAT clearance is rejected."""
    box = OrientedBoundingBox((0, 0, 0), (1, 1, 1), IDENTITY)
    with pytest.raises(AssertionError, match="Clearance"):
        box.penetration(box, clearance_m=-1e-3)


def test_zero_projection_axis_is_rejected():
    """A zero projection axis is rejected."""
    box = OrientedBoundingBox((0, 0, 0), (1, 1, 1), IDENTITY)
    with pytest.raises(AssertionError, match="non-zero"):
        box.get_bounds_along_axis((0.0, 0.0, 0.0))


def test_sampling_reproducibility_and_oriented_membership():
    """Oriented-box sampling is reproducible and remains inside the box."""
    box = OrientedBoundingBox((2.0, 3.0, 4.0), (1.0, 0.5, 0.25), _yaw(math.pi / 2))
    first = get_random_pose_within_bounding_box(box, seed=17)
    second = get_random_pose_within_bounding_box(box, seed=17)
    assert first == second
    position = torch.tensor(first.position_xyz)
    delta = position - box.center[0]
    local = torch.tensor([delta[1], -delta[0], delta[2]])
    assert (local.abs() <= box.half_extents[0] + 1e-6).all()
    assert first.rotation_xyzw == IDENTITY
