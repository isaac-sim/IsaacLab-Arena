# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check full relative poses independently of simulator lifecycle."""

import math
import torch
from types import SimpleNamespace

import pytest
from isaaclab.utils.math import quat_apply, quat_mul

from isaaclab_arena.tasks.predicates.spatial import relative_pose_matches


def _environment(subject, parent):
    poses = {"part": subject, "receiver": parent}
    return SimpleNamespace(arena_world=SimpleNamespace(get_pose_w=poses.__getitem__))


def test_relative_target_follows_rotated_parent_and_quaternion_signs():
    parent = torch.tensor([[3, -2, 0.8, math.sqrt(0.5), 0, 0, math.sqrt(0.5)]], dtype=torch.float64).repeat(2, 1)
    target_p = (0.2, -0.3, 0.1)
    target_q = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
    subject = torch.cat(
        (
            parent[:, :3] + quat_apply(parent[:, 3:], parent.new_tensor(target_p).expand(2, 3)),
            quat_mul(parent[:, 3:], parent.new_tensor(target_q).expand(2, 4)),
        ),
        dim=-1,
    )
    subject[1, 3:] *= -1
    env = _environment(subject, parent)
    assert relative_pose_matches(env, "part", "receiver", target_p, target_q, 0.001, 0.001).all()
    parent[1, 0] += 0.1
    assert relative_pose_matches(env, "part", "receiver", target_p, target_q, 0.001, 0.001).tolist() == [True, False]


def test_position_and_full_yaw_use_strict_error_thresholds():
    parent = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float64).repeat(4, 1)
    subject = parent.clone()
    subject[:, 0] = subject.new_tensor([0.125 - 1e-9, 0.125, 0.125 + 1e-9, 0])
    subject[3, 3:] = subject.new_tensor((0, 0, math.sin(0.2), math.cos(0.2)))
    env = _environment(subject, parent)
    assert relative_pose_matches(env, "part", "receiver", (0, 0, 0), (0, 0, 0, 1), 0.125, 0.3).tolist() == [
        True,
        False,
        False,
        False,
    ]
    # An exactly matched pose still fails a strict zero angular tolerance.
    subject[:] = parent
    assert not relative_pose_matches(env, "part", "receiver", (0, 0, 0), (0, 0, 0, 1), 1, 0).any()


@pytest.mark.parametrize("bad", (0, 0.5, 2, math.nan, math.inf))
def test_independently_invalid_frames_cannot_cancel(bad):
    parent = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float64).repeat(2, 1)
    subject = parent.clone()
    subject[1, 6] = bad
    parent[1, 6] = 1 / bad if bad else 0
    env = _environment(subject, parent)
    assert relative_pose_matches(env, "part", "receiver", (0, 0, 0), (0, 0, 0, 1), 0.01, 0.01).tolist() == [True, False]


def test_small_drift_is_normalized_without_changing_inputs():
    parent = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float64)
    subject = parent.clone()
    subject[:, 3:] *= 1 + 0.9e-4
    parent[:, 3:] *= 1 - 0.9e-4
    before_subject, before_parent = subject.clone(), parent.clone()
    assert relative_pose_matches(
        _environment(subject, parent), "part", "receiver", (0, 0, 0), (0, 0, 0, 1), 0.01, 0.01
    ).all()
    torch.testing.assert_close(subject, before_subject)
    torch.testing.assert_close(parent, before_parent)
