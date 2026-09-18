# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Loop-on-rod and box-containment geometry that decides whether a tool sits on its fixture."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass

from isaaclab.utils.math import quat_apply

Vec3 = tuple[float, float, float]


def _points_w(points_F: torch.Tensor, T_W_F: torch.Tensor) -> torch.Tensor:
    """Map ``(K, 3)`` points from frame ``F`` into world for every ``(N, 7)`` pose ``T_W_F``."""
    num_envs, num_points = T_W_F.shape[0], points_F.shape[0]
    q_W_F = T_W_F[:, None, 3:].expand(num_envs, num_points, 4)
    return T_W_F[:, None, :3] + quat_apply(q_W_F, points_F[None].expand(num_envs, num_points, 3))


def _as_tensor(values: tuple[Vec3, ...], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=like.device, dtype=like.dtype)


@dataclass
class Loop:
    """A circular opening in the tool frame ``T``."""

    center_xyz: Vec3
    radius_m: float

    def __post_init__(self) -> None:
        assert self.radius_m > 0.0, "Loop radius must be positive."


@dataclass
class Rod:
    """A straight segment in the fixture frame ``X``."""

    start_xyz: Vec3
    end_xyz: Vec3

    def __post_init__(self) -> None:
        assert math.dist(self.start_xyz, self.end_xyz) > 0.0, "Rod segment must be non-degenerate."


@dataclass
class LoopOnRod:
    """Success when the rod segment passes within any of the tool's loops."""

    loops: tuple[Loop, ...]
    rod: Rod

    @classmethod
    def from_dict(cls, goal: dict) -> LoopOnRod:
        """Read the ``loop``, ``rod``, and optional ``alternative_loops`` keys."""
        loops = (Loop(**goal["loop"]), *(Loop(**loop) for loop in goal.get("alternative_loops", [])))
        return cls(loops=loops, rod=Rod(**goal["rod"]))

    def evaluate(self, T_W_T: torch.Tensor, T_W_X: torch.Tensor) -> torch.Tensor:
        """Return one Boolean per environment from the tool pose ``T_W_T`` and fixture pose ``T_W_X``."""
        ends_W = _points_w(_as_tensor((self.rod.start_xyz, self.rod.end_xyz), T_W_X), T_W_X)
        start_W, direction_W = ends_W[:, :1], ends_W[:, 1:] - ends_W[:, :1]
        centers_W = _points_w(_as_tensor(tuple(loop.center_xyz for loop in self.loops), T_W_T), T_W_T)
        radii = torch.tensor([loop.radius_m for loop in self.loops], device=T_W_T.device, dtype=T_W_T.dtype)
        # Closest point of the segment to each loop center, clamped to the segment ends.
        span = torch.sum(direction_W * direction_W, dim=-1)
        fraction = (torch.sum((centers_W - start_W) * direction_W, dim=-1) / span).clamp(0.0, 1.0)
        closest_W = start_W + fraction[..., None] * direction_W
        return (torch.linalg.vector_norm(closest_W - centers_W, dim=-1) <= radii).any(dim=-1)


@dataclass
class PointInBox:
    """Success when a tool point lies in a world-aligned box translated to the fixture origin.

    The box follows the fixture's position but not its rotation, matching how the upstream
    benchmark scores its containment regions for these fixed fixtures.
    """

    minimum_xyz: Vec3
    maximum_xyz: Vec3
    point_xyz: Vec3 = (0.0, 0.0, 0.0)
    """Probe point in the tool frame ``T``; defaults to the tool origin."""

    def __post_init__(self) -> None:
        assert all(lo < hi for lo, hi in zip(self.minimum_xyz, self.maximum_xyz, strict=True)), "Box must be ordered."

    @classmethod
    def from_dict(cls, goal: dict) -> PointInBox:
        """Read the ``containment`` mapping."""
        return cls(**goal["containment"])

    def evaluate(self, T_W_T: torch.Tensor, T_W_X: torch.Tensor) -> torch.Tensor:
        """Return one Boolean per environment from the tool pose ``T_W_T`` and fixture pose ``T_W_X``."""
        offset = _points_w(_as_tensor((self.point_xyz,), T_W_T), T_W_T)[:, 0] - T_W_X[:, :3]
        minimum, maximum = _as_tensor((self.minimum_xyz,), offset)[0], _as_tensor((self.maximum_xyz,), offset)[0]
        return ((offset >= minimum) & (offset <= maximum)).all(dim=-1)


def goal_geometry_from_dict(goal: dict) -> LoopOnRod | PointInBox:
    """Pick the geometry declared by one YAML goal."""
    assert ("loop" in goal) != ("containment" in goal), "A goal declares either loop/rod or containment geometry."
    return PointInBox.from_dict(goal) if "containment" in goal else LoopOnRod.from_dict(goal)
