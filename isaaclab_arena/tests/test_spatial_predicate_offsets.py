# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check shared spatial predicates with local offsets and non-Z target axes."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_spatial_predicate_offsets(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    import pytest
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

    from isaaclab_arena.tasks.predicates.spatial import (
        _relative_axial_distances,
        depth_in_range,
        lateral_in_proximity,
        tilt_axis_aligned,
    )

    poses = {
        "subject": torch.tensor(
            [
                [0.125, 0, 0.25, 0, 0, 0, 1],
                [0.125, 0, 0.5, 0, 0, 0, 1],
                [0.25, 0, 0.75, 0, 0, 0, 1],
                [0, 0, -0.25, 0, 0, 0, 1],
            ],
            dtype=torch.float64,
        ),
        "receiver": torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float64).repeat(4, 1),
    }
    env = SimpleNamespace(num_envs=4, arena_world=SimpleNamespace(get_pose_w=poses.__getitem__))
    params = {"subject_name": "subject", "receiver_name": "receiver", "target_offset_xyz": (0, 0, 0)}
    assert depth_in_range(env, **params, depth_min=0.25, depth_max=0.5).tolist() == [True, True, False, False]
    assert depth_in_range(env, **params, depth_min=0.25, depth_max=None).tolist() == [True, True, True, False]
    assert lateral_in_proximity(env, **params, tolerance_lateral=0.125).tolist() == [True, True, False, True]
    assert tilt_axis_aligned(env, "subject", "receiver", 0.01).all()
    assert not tilt_axis_aligned(env, "subject", "receiver", 0.01, subject_axis=(0, 0, -2)).any()
    assert tilt_axis_aligned(env, "subject", "receiver", 0.01, subject_axis=(0, 0, -2), allow_antiparallel=True).all()
    assert not tilt_axis_aligned(
        env, "subject", "receiver", 0.01, subject_axis=(1, 0, 0), allow_antiparallel=True
    ).any()
    with pytest.raises(AssertionError):
        depth_in_range(env, **params, depth_min=0.5, depth_max=0.25)
    with pytest.raises(AssertionError):
        lateral_in_proximity(env, **params, tolerance_lateral=0.125, receiver_axis=(0, 0, 0))

    angles = torch.tensor([0, 0.4, -0.7, 1.1], dtype=torch.float64)
    q_W_R = quat_from_euler_xyz(angles, -angles, angles * 0.5)
    q_W_S = quat_from_euler_xyz(-angles * 0.3, angles + 0.2, -angles)
    positions_W = torch.tensor([[0.2, 0.3, 0.5], [1, 2, 3], [-1, 0, 0.8], [0, 0, 0]], dtype=torch.float64)
    poses["receiver"] = torch.cat((positions_W, q_W_R), dim=-1)
    depth = torch.tensor([-0.01, 0.015, 0.025, 0.04], dtype=torch.float64)
    lateral = torch.tensor([0, 0.0005, 0.006, 0.001], dtype=torch.float64)
    subject_offset = torch.tensor([0.04, 0.02, 0.01], dtype=torch.float64)
    target_offset = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64)
    for receiver_axis in ((2, 0, 0), (0, 0, -3), (1, 2, 3)):
        axis_R = torch.tensor(receiver_axis, dtype=torch.float64)
        axis_R /= torch.linalg.vector_norm(axis_R)
        tangent_R = torch.linalg.cross(axis_R, torch.tensor([0, 1, 0], dtype=torch.float64))
        tangent_R /= torch.linalg.vector_norm(tangent_R)
        point_R = target_offset + depth[:, None] * axis_R + lateral[:, None] * tangent_R
        point_W = positions_W + quat_apply(q_W_R, point_R)
        subject_position_W = point_W - quat_apply(q_W_S, subject_offset.expand(4, -1))
        poses["subject"] = torch.cat((subject_position_W, q_W_S), dim=-1)
        params.update(
            target_offset_xyz=tuple(target_offset.tolist()),
            subject_offset_xyz=tuple(subject_offset.tolist()),
            receiver_axis=receiver_axis,
        )
        actual_depth, actual_lateral = _relative_axial_distances(env, **params)
        torch.testing.assert_close(actual_depth, depth)
        torch.testing.assert_close(actual_lateral, lateral)
        assert depth_in_range(env, **params, depth_min=0.01, depth_max=0.03).tolist() == [False, True, True, False]
        assert depth_in_range(env, **params, depth_min=0.01, depth_max=None).tolist() == [False, True, True, True]
        assert lateral_in_proximity(env, **params, tolerance_lateral=0.004).tolist() == [True, True, False, True]
    return True


def test_spatial_predicate_offsets() -> None:
    assert run_function_with_persistent_simulation_app(_test_spatial_predicate_offsets)


def _test_default_axis_spatial_predicate_compatibility(_simulation_app) -> bool:
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

    from isaaclab_arena.tasks.predicates.spatial import depth_in_range, lateral_in_proximity, tilt_axis_aligned

    target = (0.02, -0.03, 0.01)
    params = {"subject_name": "gear", "receiver_name": "plate", "target_offset_xyz": target}
    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            relative_position = torch.tensor(
                [
                    [0, 0, 0],
                    [0.015 - 1.0e-6, 0, -0.01 + 1.0e-6],
                    [0.015, 0, 0.01],
                    [0.015 + 1.0e-6, 0, 0.01 + 1.0e-6],
                    [0.01, 0.01, -0.01 - 1.0e-6],
                    [0, 0, 0],
                ],
                device=device,
                dtype=dtype,
            )
            angles = torch.tensor([0, 0.4, -0.7, 1.1, -0.2, 0.6], device=device, dtype=dtype)
            tilt = torch.tensor([0, 14.999, 15, 15.001, 45, 180], device=device, dtype=dtype) * math.pi / 180
            q_W_P = quat_from_euler_xyz(angles, -angles, angles * 0.5)
            q_P_G = quat_from_euler_xyz(tilt, torch.zeros_like(tilt), torch.zeros_like(tilt))
            q_W_G = quat_mul(q_W_P, q_P_G)
            t_W_P = torch.tensor([0.3, -0.4, 0.8], device=device, dtype=dtype).expand(6, -1)
            target_P = torch.tensor(target, device=device, dtype=dtype)
            t_W_G = t_W_P + quat_apply(q_W_P, relative_position + target_P)
            poses = {"plate": torch.cat((t_W_P, q_W_P), dim=-1), "gear": torch.cat((t_W_G, q_W_G), dim=-1)}
            env = SimpleNamespace(num_envs=6, arena_world=SimpleNamespace(get_pose_w=poses.__getitem__))
            position_P = quat_apply_inverse(q_W_P, t_W_G - t_W_P) - target_P
            axis = torch.tensor([0, 0, 1], device=device, dtype=dtype).expand(6, -1)
            expected = [
                torch.linalg.vector_norm(position_P[:, :2], dim=-1) <= 0.015,
                (position_P[:, 2] >= -0.01) & (position_P[:, 2] <= 0.01),
                torch.sum(quat_apply(q_W_G, axis) * quat_apply(q_W_P, axis), dim=-1) >= math.cos(math.radians(15)),
            ]
            actual = [
                lateral_in_proximity(env, **params, tolerance_lateral=0.015),
                depth_in_range(env, **params, depth_min=-0.01, depth_max=0.01),
                tilt_axis_aligned(env, "gear", "plate", max_tilt_rad=math.radians(15)),
            ]
            for actual_result, expected_result in zip(actual, expected, strict=True):
                assert torch.equal(actual_result, expected_result), (device, dtype)
            assert not expected[2][-1]
    return True


def test_default_axis_spatial_predicate_compatibility() -> None:
    assert run_function_with_persistent_simulation_app(_test_default_axis_spatial_predicate_compatibility)
