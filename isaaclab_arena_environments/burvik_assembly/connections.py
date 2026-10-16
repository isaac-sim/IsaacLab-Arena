# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Seated push-fit retention, implemented as resettable part-to-part PhysX joints."""

import math
import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, subtract_frame_transforms
from pxr import Gf, UsdPhysics


def reset_workbench(env, env_ids):
    """Restore the kinematic workbench without writing unsupported dynamic velocities."""
    body = env.scene["workbench"]
    pose = body.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    body.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def attachment_errors(env, part_name, position, quaternion):
    """Measure position and orientation errors against the live tray frame."""
    T_W_T = env.arena_world.get_pose_w("tray")
    T_W_P = env.arena_world.get_pose_w(part_name)
    t_T_P = torch.tensor(position, device=env.device).expand(env.num_envs, -1)
    q_T_P = torch.tensor(quaternion, device=env.device).expand(env.num_envs, -1)
    target_p, target_q = combine_frame_transforms(T_W_T[:, :3], T_W_T[:, 3:], t_T_P, q_T_P)
    distance = torch.linalg.vector_norm(T_W_P[:, :3] - target_p, dim=-1)
    if part_name.startswith("leg_"):
        # Smooth round pins have no keyed axial rotation. Test their insertion axes.
        z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
        dot = (quat_apply(T_W_P[:, 3:], z) * quat_apply(target_q, z)).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
    else:
        dot = torch.sum(T_W_P[:, 3:] * target_q, dim=-1).abs().clamp(max=1.0)
        angle = 2.0 * torch.acos(dot)
    return distance, angle


def joint_path(env, env_id, part_name):
    return f"{env.scene.env_prim_paths[env_id]}/BurvikConnections/{part_name}"


def attachment_enabled(env, part_name):
    """Read real USD joint state; pose alignment alone never implies attachment."""
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for index in range(env.num_envs):
        joint = UsdPhysics.FixedJoint.Get(env.sim.stage, joint_path(env, index, part_name))
        if joint:
            result[index] = bool(joint.GetJointEnabledAttr().Get())
    return result


def clear_connections(env, env_ids, attachments):
    """Remove only the resetting environments' retention joints."""
    if hasattr(env, "burvik_motion_measurement"):
        env.burvik_motion_measurement.reset(env_ids)
    ids = range(env.num_envs) if env_ids is None else env_ids
    for index in ids:
        for name in attachments:
            path = joint_path(env, int(index), name)
            if env.sim.stage.GetPrimAtPath(path):
                env.sim.stage.RemovePrim(path)


def engage_seated_connections(env, env_ids, attachments):
    """Latch seated pins with 0.3 mm radial tolerance; allow 1.3 mm axial error only for legs."""
    for name, transform in attachments.items():
        unattached = ~attachment_enabled(env, name)
        if not bool(unattached.any()):
            continue
        distance, angle = attachment_errors(env, name, **transform)
        tray = env.arena_world.get_pose_w("tray")
        part = env.arena_world.get_pose_w(name)
        nominal_p = torch.tensor(transform["position"], device=env.device).expand(env.num_envs, -1)
        nominal_q = torch.tensor(transform["quaternion"], device=env.device).expand(env.num_envs, -1)
        target_p, target_q = combine_frame_transforms(tray[:, :3], tray[:, 3:], nominal_p, nominal_q)
        local_error = quat_apply_inverse(target_q, part[:, :3] - target_p)
        radial_error = torch.linalg.vector_norm(local_error[:, :2], dim=-1)
        position_tolerance = 0.0003 if name == "handle" else 0.0013
        ready = (distance < position_tolerance) & (radial_error < 0.0003) & (angle < math.radians(1.0)) & unattached
        for index in ready.nonzero(as_tuple=False).flatten().tolist():
            T_W_T = env.arena_world.get_pose_w("tray")[index]
            T_W_P = env.arena_world.get_pose_w(name)[index]
            position, quaternion = subtract_frame_transforms(T_W_T[:3], T_W_T[3:], T_W_P[:3], T_W_P[3:])
            joint = UsdPhysics.FixedJoint.Define(env.sim.stage, joint_path(env, index, name))
            root = env.scene.env_prim_paths[index]
            joint.CreateBody0Rel().SetTargets([f"{root}/tray"])
            joint.CreateBody1Rel().SetTargets([f"{root}/{name}"])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*position.tolist()))
            x, y, z, w = quaternion.tolist()
            joint.CreateLocalRot0Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.CreateJointEnabledAttr(True)
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateCollisionEnabledAttr(False)
            # Ideal seated retention: fracture strengths are outside this simplified push-fit model.
