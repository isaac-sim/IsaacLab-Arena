# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Logic tests for the collision-free IK path, with cuRobo's solver faked out.

Exercises what ``solve_ik_feasibility`` does around cuRobo when collision checking is on -- mute the
hand links for the solve, restore them afterwards, and fold cuRobo's ``success`` into the verdict --
against a fake kinematics config that mimics cuRobo's in-place sphere edits. Needs no GPU, but does
need the cuRobo image, since the module under test imports cuRobo at import time.
"""

from __future__ import annotations

import torch

import pytest

pytestmark = pytest.mark.curobo_deps

HAND_LINKS = ["left_finger", "right_finger"]
ARM_LINK = "forearm"
DISABLED_RADIUS = -100.0
"""Radius cuRobo writes into a muted sphere; anything negative is ignored by its collision kernels."""


class _FakeKinematicsConfig:
    """Stand-in for cuRobo's ``KinematicsTensorConfig``, one unit-radius sphere per link."""

    def __init__(self, link_names: list[str]) -> None:
        self.link_name_to_idx_map = {name: index for index, name in enumerate(link_names)}
        self._spheres = {name: torch.tensor([[0.0, 0.0, 0.0, 0.05]]) for name in link_names}

    def get_link_spheres(self, link_name: str) -> torch.Tensor:
        return self._spheres[link_name]

    def update_link_spheres(self, link_name: str, sphere_position_radius: torch.Tensor) -> None:
        self._spheres[link_name] = sphere_position_radius

    def disable_link_spheres(self, link_name: str) -> None:
        spheres = self.get_link_spheres(link_name).clone()
        spheres[:, 3] = DISABLED_RADIUS
        self.update_link_spheres(link_name, spheres)

    def radius(self, link_name: str) -> float:
        return float(self._spheres[link_name][0, 3].item())


class _FakeIKResult:
    def __init__(self, num_poses: int, success: bool) -> None:
        self.position_error = torch.zeros(num_poses)
        self.rotation_error = torch.zeros(num_poses)
        self.success = torch.full((num_poses,), success, dtype=torch.bool)


class _FakeIKSolver:
    """Records the sphere radii cuRobo would have seen at solve time, then returns a canned result."""

    def __init__(self, kinematics_config: _FakeKinematicsConfig, success: bool, raises: bool = False) -> None:
        self.kinematics = type("_Kinematics", (), {"kinematics_config": kinematics_config})()
        self._success = success
        self._raises = raises
        self.radii_during_solve: dict[str, float] = {}

    def solve_batch(self, goal_pose, seed_config=None) -> _FakeIKResult:
        config = self.kinematics.kinematics_config
        self.radii_during_solve = {name: config.radius(name) for name in config.link_name_to_idx_map}
        if self._raises:
            raise RuntimeError("solve blew up")
        return _FakeIKResult(goal_pose.position.shape[0], self._success)


class _FakeHost:
    """Minimal ``CuroboIKSolver``-shaped host: owns the solver, the hand links, and the pose plumbing."""

    def __init__(self, ik_solver: _FakeIKSolver, hand_link_names: list[str]) -> None:
        import logging

        self.ik_solver = ik_solver
        self.hand_link_names = hand_link_names
        self.logger = logging.getLogger("test_ik_solver_utils")

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=torch.float32)

    def _make_pose(self, position, quaternion, quat_is_xyzw: bool = True):
        return type("_Pose", (), {"position": position, "quaternion": quaternion})()


def _make_host(success: bool = True, raises: bool = False) -> tuple[_FakeHost, _FakeKinematicsConfig]:
    """Build a fake host whose solve reports ``success`` (cuRobo's converged-and-collision-free flag)."""
    kinematics_config = _FakeKinematicsConfig([*HAND_LINKS, ARM_LINK])
    host = _FakeHost(_FakeIKSolver(kinematics_config, success=success, raises=raises), HAND_LINKS)
    return host, kinematics_config


def _identity_grasps(num_poses: int = 2) -> torch.Tensor:
    return torch.eye(4).repeat(num_poses, 1, 1)


def test_collision_free_solve_mutes_only_the_hand_links():
    """The gripper's links are muted for the solve; the rest of the arm keeps its spheres."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, _ = _make_host()
    solve_ik_feasibility(host, _identity_grasps(), require_collision_free=True)

    radii = host.ik_solver.radii_during_solve
    assert [radii[name] for name in HAND_LINKS] == [DISABLED_RADIUS, DISABLED_RADIUS]
    assert radii[ARM_LINK] > 0.0


def test_hand_links_are_restored_after_the_solve():
    """Muting is scoped to the solve, so a later solve (or planner) sees the original spheres."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, kinematics_config = _make_host()
    solve_ik_feasibility(host, _identity_grasps(), require_collision_free=True)

    assert all(kinematics_config.radius(name) > 0.0 for name in HAND_LINKS)


def test_hand_links_are_restored_when_the_solve_raises():
    """A failed solve must not leave the shared kinematics permanently muted."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, kinematics_config = _make_host(raises=True)
    with pytest.raises(RuntimeError):
        solve_ik_feasibility(host, _identity_grasps(), require_collision_free=True)

    assert all(kinematics_config.radius(name) > 0.0 for name in HAND_LINKS)


def test_reachability_only_solve_leaves_every_sphere_alone():
    """With collision checking off, the solve is pure reachability and touches no spheres."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, _ = _make_host()
    solve_ik_feasibility(host, _identity_grasps(), require_collision_free=False)

    assert all(radius > 0.0 for radius in host.ik_solver.radii_during_solve.values())


def test_colliding_pose_is_infeasible_only_when_collision_checking_is_on():
    """A converged but colliding solution passes the reachability-only check and fails the collision one."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, _ = _make_host(success=False)
    reachable, _, _ = solve_ik_feasibility(host, _identity_grasps(), require_collision_free=False)
    collision_free, _, _ = solve_ik_feasibility(host, _identity_grasps(), require_collision_free=True)

    assert reachable.tolist() == [True, True]
    assert collision_free.tolist() == [False, False]


def test_unknown_hand_link_is_reported_against_the_robot_config():
    """A hand link with no spheres in the robot config fails loudly rather than silently doing nothing."""
    from isaaclab_arena_curobo.utils.ik_solver_utils import solve_ik_feasibility

    host, _ = _make_host()
    host.hand_link_names = ["not_a_link"]
    with pytest.raises(AssertionError, match="not_a_link"):
        solve_ik_feasibility(host, _identity_grasps(), require_collision_free=True)
