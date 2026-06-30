# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The Maple exterior_cam agentview quaternion must look from eye toward target (xyzw, this Isaac Lab)."""


def test_agentview_quat_is_xyzw_and_looks_at_target():
    # Deferred imports: maple_cameras pulls in Isaac Lab configclasses/math (needs the booted app).
    import torch

    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_arena_environments.maple_cameras import _CAM_EYE, _CAM_TARGET, _agentview_ros_quat_xyzw

    q_xyzw = _agentview_ros_quat_xyzw(_CAM_EYE, _CAM_TARGET)
    assert len(q_xyzw) == 4
    # Unit quaternion.
    assert abs(sum(v * v for v in q_xyzw) ** 0.5 - 1.0) < 1e-5

    # This Isaac Lab's matrix_from_quat is xyzw (matrix_from_quat((0,0,0,1)) == identity), matching the
    # xyzw quat the helper returns. The camera's optical +Z (OpenCV/ROS forward, column 2 of
    # R_cam_to_world) must point from eye toward target.
    rot = matrix_from_quat(torch.tensor([q_xyzw], dtype=torch.float32))[0]
    cam_forward = rot[:, 2]
    want = torch.tensor(_CAM_TARGET, dtype=torch.float32) - torch.tensor(_CAM_EYE, dtype=torch.float32)
    want = want / torch.linalg.norm(want)
    assert float(cam_forward @ want) > 0.999, f"camera forward {cam_forward.tolist()} does not look at target"

    # Lock the xyzw schema on a NONTRIVIAL rotation: this agentview is far from identity, and round-tripping
    # the quat through quat_from_matrix(matrix_from_quat(q)) in xyzw must reproduce it. A wxyz mislabel would
    # both fail the look-at above and round-trip to a different quaternion.
    from isaaclab.utils.math import quat_from_matrix

    assert abs(q_xyzw[3]) < 0.999, f"expected a nontrivial rotation, got near-identity {q_xyzw}"
    q_roundtrip = quat_from_matrix(rot.unsqueeze(0))[0]
    # Quaternions are equal up to sign; compare with the sign aligned.
    sign = 1.0 if float((torch.tensor(q_xyzw) * q_roundtrip).sum()) >= 0 else -1.0
    assert torch.allclose(torch.tensor(q_xyzw, dtype=torch.float32), sign * q_roundtrip, atol=1e-5)
