# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP2 terms of Arena components.

Terms receive the scene data they read as inputs and write their outputs in place, so that Isaac Lab's MDP2 builder
can merge them into a step that replays as one CUDA graph. They run as Torch hooks between MDP2's generated kernels.
"""

from __future__ import annotations

import math
import torch

import isaaclab.utils.math as math_utils
from isaaclab_experimental.mdp2 import State, Term, TermContext
from isaaclab_experimental.mdp2.rng import uniform

# Observations.

end_effector_pos_w = Term(
    torch=lambda body_link_pose_w, body_ids: body_link_pose_w[:, body_ids][:, 0, :3], shape=("worlds", 3)
)
"""Position of the body of ``body_ids`` (one body) in the world frame [m], like the DROID ``ee_pos`` observation."""

end_effector_quat_w = Term(
    torch=lambda body_link_pose_w, body_ids: body_link_pose_w[:, body_ids][:, 0, 3:7], shape=("worlds", 4)
)
"""Orientation ``(x, y, z, w)`` of the body of ``body_ids`` (one body) in the world frame, like ``ee_quat``."""


# Events.


def _reset_joint_targets(mask: torch.Tensor, default_joint_pos: torch.Tensor, joint_pos_target: torch.Tensor) -> None:
    joint_pos_target.copy_(torch.where(mask[:, None], default_joint_pos, joint_pos_target))


reset_joint_targets = Term(torch=_reset_joint_targets)
"""Set the joint position targets of the reset worlds to the default joint positions, as
``reset_scene_to_default(reset_joint_targets=True)`` does. Bind ``joint_pos_target`` as an output."""


def _draw_layout(mask: torch.Tensor, rng: torch.Tensor, pool: torch.Tensor, env_origins: torch.Tensor, **poses) -> None:
    """Write one pooled layout per reset world: each ``pose_<k>`` output gets columns ``7k:7k+7`` of the layout."""
    index = uniform(rng, 0.0, float(pool.shape[0])).long().clamp(max=pool.shape[0] - 1)
    layout = pool[index]
    reset = mask[:, None]
    for name, pose in poses.items():
        if not name.startswith("pose_"):
            continue
        k = int(name.removeprefix("pose_"))
        T_W_O = layout[:, 7 * k : 7 * k + 7].clone()
        T_W_O[:, :3] += env_origins
        pose.copy_(torch.where(reset, T_W_O, pose))
        velocity = poses[f"vel_{k}"]
        velocity.copy_(torch.where(reset, torch.zeros_like(velocity), velocity))


def make_layout_pool_term(layouts: torch.Tensor) -> Term:
    """Return a reset event term that draws relation-solved layouts from a pool, at random for each reset world.

    Args:
        layouts: Solved layouts, one row per layout and ``(x, y, z, qx, qy, qz, qw)`` environment-local root poses of
            the placed assets side by side. They fill the term's pool once, at startup.

    Returns:
        An event term with outputs ``pose_<k>`` and ``vel_<k>``, the root pose and velocity of placed asset ``k``, and
        the input ``env_origins``.
    """
    assert layouts.ndim == 2 and layouts.shape[1] % 7 == 0, "Layouts must hold 7 pose values per placed asset"
    host_layouts = layouts.detach().to("cpu", torch.float32)

    def fill_pool(ctx: TermContext, **_) -> None:
        ctx.state["pool"].copy_(host_layouts.to(ctx.state["pool"].device))

    return Term(
        torch=_draw_layout,
        state={"pool": State(shape=(host_layouts.shape[0], host_layouts.shape[1]), persistent=True)},
        random=True,
        init=fill_pool,
    )


# Conditions of success and metrics.


def _z_axis(quat: torch.Tensor) -> torch.Tensor:
    """Return the z axis of ``(x, y, z, w)`` orientations, without constants from the host."""
    x, y, z, w = quat.unbind(-1)
    return torch.stack((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1)


def _seated_at_target(
    subject_pose_w: torch.Tensor,
    subject_vel_w: torch.Tensor,
    receiver_pose_w: torch.Tensor,
    *,
    target_offset: tuple[float, float, float],
    max_lateral: float,
    max_depth_below: float,
    max_height_above: float,
    max_tilt_rad: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> torch.Tensor:
    q_W_R = receiver_pose_w[:, 3:7]
    t_R_T = torch.zeros_like(receiver_pose_w[:, :3])
    for axis, value in enumerate(target_offset):
        t_R_T[:, axis] = value
    t_W_T = receiver_pose_w[:, :3] + math_utils.quat_apply(q_W_R, t_R_T)
    # The subject origin relative to the target, in the target frame, which has the receiver's orientation.
    t_T_S = math_utils.quat_apply_inverse(q_W_R, subject_pose_w[:, :3] - t_W_T)
    lateral = t_T_S[:, :2].norm(dim=-1) <= max_lateral
    depth = (t_T_S[:, 2] >= -max_depth_below) & (t_T_S[:, 2] <= max_height_above)
    upright = (_z_axis(subject_pose_w[:, 3:7]) * _z_axis(q_W_R)).sum(-1) >= math.cos(max_tilt_rad)
    linear_still = subject_vel_w[:, :3].norm(dim=-1) <= max_linear_speed
    angular_still = subject_vel_w[:, 3:6].norm(dim=-1) <= max_angular_speed
    return (lateral & depth & upright & linear_still & angular_still).float()


seated_at_target = Term(torch=_seated_at_target)
"""One where the subject sits at a target point of the receiver: within ``max_lateral`` [m] of it across the
receiver's z axis, between ``max_depth_below`` below and ``max_height_above`` above it along that axis [m], tilted
less than ``max_tilt_rad`` from the receiver's z axis, and slower than ``max_linear_speed`` [m/s] and
``max_angular_speed`` [rad/s]. The target is ``target_offset`` in the receiver frame [m]. Together, these are
the ``lateral_in_proximity``, ``depth_in_range``, ``tilt_axis_aligned``, and ``velocity_below_threshold`` predicates."""


def _held_for(condition: torch.Tensor, count: torch.Tensor, steps: int) -> torch.Tensor:
    count.copy_(torch.where(condition > 0.0, count + 1.0, torch.zeros_like(count)))
    return (count >= steps).float()


held_for = Term(torch=_held_for, state={"count": State()})
"""One once ``condition`` held for ``steps`` consecutive control steps, like ``TrueForConsecutiveStepsCfg``. Its state
``count`` holds the current run of steps."""


def _moved(root_com_vel_w: torch.Tensor, moved: torch.Tensor, speed: float) -> torch.Tensor:
    moved.logical_or_(root_com_vel_w[:, :3].norm(dim=1) > speed)
    return moved.clone()


moved = Term(torch=_moved, state={"moved": State()})
"""One once the object moved faster than ``speed`` [m/s] in the episode, like the object-moved-rate metric."""
