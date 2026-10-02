# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PhysX retention for aligned, robot-inserted service components.

The fixed constraints approximate detent retention after insertion. They never
write object poses. A pressed release remains disengaged until the part leaves
the socket, allowing a single arm to release and then extract the component.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from .measurements import Bounds, box_contained, point_velocity


@dataclass(frozen=True)
class Socket:
    """Specify a retained component pose in its parent's coordinate frame."""

    name: str
    parent: str
    candidates: tuple[str, ...]
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    position_tolerance_m: float = 0.003
    orientation_tolerance_deg: float = 5.0
    capture_speed_m_s: float = 0.05
    capture_angular_speed_rad_s: float = 0.5
    capture_bounds: Bounds | None = None
    """Optional parent-frame volume that must contain the candidate's seating feature."""

    candidate_bounds: dict[str, Bounds] = field(default_factory=dict)
    """Candidate-local seating-feature bounds, required for every candidate when capture_bounds is set."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.position_tolerance_m) and self.position_tolerance_m > 0.0
        if self.capture_bounds is None:
            assert not self.candidate_bounds, "Candidate seating bounds require a parent capture volume."
        else:
            assert set(self.candidate_bounds) == set(self.candidates), "Provide seating bounds for every candidate."


def relative_pose(T_W_A: torch.Tensor, T_W_B: torch.Tensor) -> torch.Tensor:
    """Return T_A_B from poses expressed in world coordinates, using XYZW quaternions."""
    q_A_W = quat_conjugate(T_W_A[:, 3:])
    t_A_B = quat_apply(q_A_W, T_W_B[:, :3] - T_W_A[:, :3])
    q_A_B = quat_mul(q_A_W, T_W_B[:, 3:])
    return torch.cat((t_A_B, q_A_B), dim=-1)


def environment_prim_path(env, name: str, env_path: str) -> str:
    """Resolve a scene asset's configured namespace to one concrete environment."""
    path = env.scene[name].cfg.prim_path
    return path.replace("{ENV_REGEX_NS}", env_path).replace(env.scene.env_regex_ns, env_path)


class RetainedSocket:
    """Own physical retention constraints and independent release state per environment."""

    def __init__(self, env, socket: Socket) -> None:
        self.env = env
        self.socket = socket
        self.attached: list[str | None] = [None] * env.num_envs
        self._released: list[str | None] = [None] * env.num_envs
        self._previous_pressed = [False] * env.num_envs
        self._joints: list[dict[str, UsdPhysics.FixedJoint]] = []
        self._capture_bounds = None
        self._candidate_bounds = {}
        if socket.capture_bounds is not None:
            self._capture_bounds = torch.as_tensor(socket.capture_bounds, device=env.device, dtype=torch.float32)
            for candidate, bounds in socket.candidate_bounds.items():
                self._candidate_bounds[candidate] = torch.as_tensor(bounds, device=env.device, dtype=torch.float32)
        stage = env.sim.stage
        for env_path in env.scene.env_prim_paths:
            joints = {}
            container = f"{env_path}/service_connectors"
            UsdGeom.Scope.Define(stage, container)
            parent_path = environment_prim_path(env, socket.parent, env_path)
            for candidate in socket.candidates:
                child_path = environment_prim_path(env, candidate, env_path)
                assert stage.GetPrimAtPath(parent_path).HasAPI(UsdPhysics.RigidBodyAPI), parent_path
                assert stage.GetPrimAtPath(child_path).HasAPI(UsdPhysics.RigidBodyAPI), child_path
                joint = UsdPhysics.FixedJoint.Define(stage, f"{container}/{socket.name}_{candidate}")
                joint.CreateJointEnabledAttr(False)
                joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
                joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
                joint.CreateLocalPos0Attr(Gf.Vec3f(*socket.position_xyz))
                q = socket.rotation_xyzw
                joint.CreateLocalRot0Attr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
                joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
                joint.CreateExcludeFromArticulationAttr(True)
                joint.CreateCollisionEnabledAttr(False)
                joints[candidate] = joint
            self._joints.append(joints)

    def reset(self, env_ids: list[int], initial: str | None) -> None:
        """Restore connector state after ordinary scene reset has restored the bodies."""
        assert initial is None or initial in self.socket.candidates
        for env_id in env_ids:
            for candidate, joint in self._joints[env_id].items():
                joint.GetJointEnabledAttr().Set(candidate == initial)
            self.attached[env_id] = initial
            self._released[env_id] = None
            self._previous_pressed[env_id] = False

    def detach(self, env_id: int, require_extraction: bool = True) -> None:
        """Disengage retention without applying force or moving the component."""
        candidate = self.attached[env_id]
        if candidate is not None:
            self._joints[env_id][candidate].GetJointEnabledAttr().Set(False)
            self._released[env_id] = candidate if require_extraction else None
            self.attached[env_id] = None

    def candidate_errors(self, candidate: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Measure positional, angular, and relative-speed capture errors."""
        world = self.env.arena_world
        T_W_P = world.get_pose_w(self.socket.parent)
        T_W_C = world.get_pose_w(candidate)
        T_P_C = relative_pose(T_W_P, T_W_C)
        target_t = T_P_C.new_tensor(self.socket.position_xyz)
        target_q = T_P_C.new_tensor(self.socket.rotation_xyzw)
        distance = torch.linalg.vector_norm(T_P_C[:, :3] - target_t, dim=-1)
        dot = torch.sum(T_P_C[:, 3:] * target_q, dim=-1).abs().clamp(max=1.0)
        angle = 2.0 * torch.acos(dot)
        omega_W_P = world.get_root_angular_velocity_w(self.socket.parent)
        # ArenaWorld returns COM velocity; capture compares the candidate root
        # with the authored socket point S, transporting both from their COMs.
        t_W_S = T_W_P[:, :3] + quat_apply(T_W_P[:, 3:], target_t.expand_as(omega_W_P))
        socket_velocity = point_velocity(
            world.get_root_linear_velocity_w(self.socket.parent),
            omega_W_P,
            t_W_S,
            self.env.scene[self.socket.parent].data.root_com_pos_w.torch,
        )
        candidate_velocity = point_velocity(
            world.get_root_linear_velocity_w(candidate),
            world.get_root_angular_velocity_w(candidate),
            T_W_C[:, :3],
            self.env.scene[candidate].data.root_com_pos_w.torch,
        )
        relative_velocity = candidate_velocity - socket_velocity
        speed = torch.linalg.vector_norm(relative_velocity, dim=-1)
        return distance, angle, speed

    def seated_candidates(self, require_stationary: bool = True) -> list[str | None]:
        """Identify aligned components, optionally requiring stable insertion before capture."""
        seated: list[str | None] = [None] * self.env.num_envs
        for candidate in self.socket.candidates:
            distance, angle, speed = self.candidate_errors(candidate)
            world = self.env.arena_world
            angular_speed = torch.linalg.vector_norm(
                world.get_root_angular_velocity_w(candidate) - world.get_root_angular_velocity_w(self.socket.parent),
                dim=-1,
            )
            aligned = (distance <= self.socket.position_tolerance_m) & (
                angle <= math.radians(self.socket.orientation_tolerance_deg)
            )
            if self._capture_bounds is not None:
                T_P_C = relative_pose(world.get_pose_w(self.socket.parent), world.get_pose_w(candidate))
                aligned &= box_contained(T_P_C, self._candidate_bounds[candidate], self._capture_bounds)
            if require_stationary:
                aligned &= (speed <= self.socket.capture_speed_m_s) & (
                    angular_speed <= self.socket.capture_angular_speed_rad_s
                )
            aligned = aligned.tolist()
            for env_id, matches in enumerate(aligned):
                if matches:
                    assert seated[env_id] is None, "Two components cannot occupy the same socket."
                    seated[env_id] = candidate
        return seated

    def update(self, pressed: list[bool], capture_allowed: list[bool] | None = None) -> None:
        """Release on button edges and retain components only after physical insertion."""
        allowed = [True] * self.env.num_envs if capture_allowed is None else capture_allowed
        seated = self.seated_candidates()
        distances = {}
        for candidate in self.socket.candidates:
            distances[candidate] = self.candidate_errors(candidate)[0].tolist()
        for env_id in range(self.env.num_envs):
            if pressed[env_id] and not self._previous_pressed[env_id]:
                self.detach(env_id)
            self._previous_pressed[env_id] = pressed[env_id]
            released = self._released[env_id]
            if released is not None and distances[released][env_id] > 0.02:
                self._released[env_id] = None
            if self.attached[env_id] is not None:
                continue
            if pressed[env_id] or not allowed[env_id] or self._released[env_id] is not None:
                continue
            candidate = seated[env_id]
            if candidate is not None:
                sensor = self.env.scene[f"service_contact_{self.socket.name}_{candidate}"]
                forces = sensor.data.normal_force_matrix_w
                assert forces is not None, "Retention requires a filtered component contact sensor."
                contact_force = torch.linalg.vector_norm(forces.torch[env_id], dim=-1).max().item()
                if contact_force <= 0.01:
                    continue
                self._joints[env_id][candidate].GetJointEnabledAttr().Set(True)
                self.attached[env_id] = candidate
