# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only forward kinematics for reachable TCP waypoint generation."""

import numpy as np
from scipy.spatial.transform import Rotation

from pxr import Usd, UsdGeom, UsdPhysics

from .config import ARM_JOINT_NAMES
from .control import TCP_OFFSET


def _transform(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
    matrix[:3, 3] = position
    return matrix


class YamForwardKinematics:
    """Read a fixed YAM's USD joint frames without changing its simulated state."""

    def __init__(self, env, side):
        assert env.num_envs == 1, "This waypoint model supports one fixed-base YAM instance"
        root = env.sim.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/{side.title()}Robot")
        joints = {}
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joints[prim.GetName()] = UsdPhysics.RevoluteJoint(prim)
        cache = UsdGeom.XformCache()
        body = env.sim.stage.GetPrimAtPath(joints[ARM_JOINT_NAMES[0]].GetBody0Rel().GetTargets()[0])
        self.base = np.asarray(cache.GetLocalToWorldTransform(body)).T.copy()
        self.frames = []
        for name in ARM_JOINT_NAMES:
            joint = joints[name]
            parent = _transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            child = _transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            self.frames.append((parent, np.linalg.inv(child), joint.GetAxisAttr().Get().lower()))

    def tcp_pose(self, joints):
        """Return the world TCP position and scipy rotation for six joint angles in radians."""
        assert len(joints) == len(self.frames), "Supply all six YAM arm angles"
        matrix = self.base.copy()
        for angle, (parent, child_inverse, axis) in zip(joints, self.frames, strict=True):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
            matrix = matrix @ parent @ rotation @ child_inverse
        position = matrix[:3, 3] + matrix[:3, :3] @ np.asarray(TCP_OFFSET)
        return position, Rotation.from_matrix(matrix[:3, :3])
