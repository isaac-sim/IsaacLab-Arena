# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pose USD articulation geometry without spawning a simulation.

Placement geometry is computed before an Isaac Lab articulation exists, so runtime link poses are
unavailable. This USD-joint implementation also avoids requiring a separate CuRobo kinematics
configuration for every embodiment. Matrices follow USD's row-vector convention:
``point_parent = point_local @ matrix``.
"""

from __future__ import annotations

import numpy as np
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_AXIS_VECTORS: dict[str, Gf.Vec3d] = {
    "X": Gf.Vec3d(1.0, 0.0, 0.0),
    "Y": Gf.Vec3d(0.0, 1.0, 0.0),
    "Z": Gf.Vec3d(0.0, 0.0, 1.0),
}


def articulation_joint_prims(root_prim: Usd.Prim) -> dict[str, Usd.Prim]:
    """Return movable joint prims under root_prim, keyed by joint name."""
    joint_prims: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint) and not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        name = prim.GetName()
        assert name not in joint_prims, (
            f"Duplicate joint name '{name}' under {root_prim.GetPath()}:"
            f" {joint_prims[name].GetPath()} and {prim.GetPath()}."
        )
        joint_prims[name] = prim
    return joint_prims


def resolve_joint_pos_patterns(
    joint_names: Iterable[str],
    joint_pos: Mapping[str, float],
) -> dict[str, float]:
    """Expand Isaac Lab-style joint keys against the joint names an articulation actually has.

    Isaac Lab full-matches ``init_state.joint_pos`` keys as regexes, so one key such as
    ``"right_outer.*"`` sets several joints at once. Keys matching no joint are dropped rather than
    rejected, because a config may name joints only some variants of an asset carry; those joints
    stay at zero.

    Args:
        joint_names: Joint names the articulation has.
        joint_pos: Joint positions keyed by exact joint name or by regex.

    Returns:
        Joint positions keyed by exact joint name. Later keys win where patterns overlap.
    """
    names = list(joint_names)
    resolved: dict[str, float] = {}
    for pattern, value in joint_pos.items():
        for name in names:
            if name == pattern or re.fullmatch(pattern, name):
                resolved[name] = float(value)
    return resolved


def compute_posed_prim_world_deltas(
    stage: Usd.Stage,
    root_prim_path: str,
    joint_pos: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Return the world-transform delta that posing at joint_pos applies to each articulated body.

    A point already in stage world space moves to its posed location via ``point @ delta`` for the
    delta of the body it belongs to. Bodies absent from the result do not move, so a caller resolves
    a geometry prim's delta from its nearest ancestor present in the mapping.

    Joints not named in joint_pos are posed at zero. Revolute values are radians, matching Isaac
    Lab rather than USD's degrees. Prismatic values are in stage linear units, which coincide with
    metres for robot USDs authored at ``metersPerUnit = 1``.

    Args:
        stage: Stage holding the articulation.
        root_prim_path: Prim path to search for joints and bodies under.
        joint_pos: Joint positions keyed by joint name.

    Returns:
        Row-vector 4x4 world deltas keyed by body prim path.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    assert root_prim, f"No prim found at path {root_prim_path}"

    joint_prims = articulation_joint_prims(root_prim)
    unknown_joints = set(joint_pos) - set(joint_prims)
    assert not unknown_joints, (
        f"Joint positions name joints absent under {root_prim_path}: {sorted(unknown_joints)}."
        f" Available joints: {sorted(joint_prims)}."
    )

    edges = _joint_edges(root_prim, joint_prims, joint_pos)
    if not edges:
        return {}

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rest_transforms: dict[str, np.ndarray] = {}
    for path in {path for edge in edges for path in (edge.parent, edge.child) if path}:
        prim = stage.GetPrimAtPath(path)
        assert prim, f"Joint references a missing prim: {path}"
        rest_transforms[path] = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    posed_transforms = _propagate_joint_motion(edges, rest_transforms)
    return {path: np.linalg.inv(rest_transforms[path]) @ posed for path, posed in posed_transforms.items()}


def resolve_prim_world_delta(prim_path: str, body_deltas: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Return the delta of the nearest ancestor of prim_path in body_deltas, or None if unposed."""
    candidate = prim_path
    while candidate and candidate != "/":
        if candidate in body_deltas:
            return body_deltas[candidate]
        candidate = candidate.rsplit("/", 1)[0]
    return None


@dataclass(frozen=True, slots=True)
class _JointEdge:
    """A parent-to-child articulation link with the motion its joint applies."""

    parent: str
    """Prim path of the joint's body0, empty when the joint attaches to the world."""

    child: str
    """Prim path of the joint's body1."""

    motion: np.ndarray
    """Row-vector 4x4 mapping the child joint frame into the parent joint frame."""

    local_0: np.ndarray
    """Row-vector 4x4 mapping the joint frame into body0."""

    local_1: np.ndarray
    """Row-vector 4x4 mapping the joint frame into body1."""


def _joint_edges(
    root_prim: Usd.Prim,
    joint_prims: Mapping[str, Usd.Prim],
    joint_pos: Mapping[str, float],
) -> list[_JointEdge]:
    """Build the articulation's parent-to-child edges, including fixed joints that carry no motion."""
    root_sdf_path = root_prim.GetPath()
    joint_pos_by_path = {prim.GetPath().pathString: joint_pos.get(name, 0.0) for name, prim in joint_prims.items()}

    edges: list[_JointEdge] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint_path = prim.GetPath().pathString
        parent, child = _joint_body_paths(prim)
        # Compare path components: a plain prefix test would also match a sibling root's children.
        if child is None or not Sdf.Path(child).HasPrefix(root_sdf_path):
            continue
        if joint_path in joint_pos_by_path:
            motion = _joint_motion(prim, joint_pos_by_path[joint_path])
        else:
            motion = np.eye(4)
        edges.append(
            _JointEdge(
                parent=parent if parent is not None else "",
                child=child,
                motion=motion,
                local_0=_joint_local_frame(prim, index=0),
                local_1=_joint_local_frame(prim, index=1),
            )
        )
    return edges


def _propagate_joint_motion(
    edges: list[_JointEdge],
    rest_transforms: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Walk the articulation outward from its roots, composing each joint's motion."""
    edges_by_parent: dict[str, list[_JointEdge]] = {}
    for edge in edges:
        edges_by_parent.setdefault(edge.parent, []).append(edge)

    child_paths = {edge.child for edge in edges}
    root_paths = [edge.parent for edge in edges if edge.parent not in child_paths]
    assert root_paths, "articulation has no root body: every body is a joint child, so it is fully cyclic"

    posed: dict[str, np.ndarray] = {}
    queue: list[tuple[str, np.ndarray]] = []
    for path in dict.fromkeys(root_paths):
        # A world-attached joint has no body0 prim, so its parent frame is the stage origin.
        parent_world = rest_transforms[path] if path else np.eye(4)
        if path:
            posed[path] = parent_world
        queue.append((path, parent_world))

    while queue:
        parent_path, parent_world = queue.pop()
        for edge in edges_by_parent.get(parent_path, ()):
            # Closed loops reach a body through a second joint: Agibot's gripper four-bars, Galbot's
            # fixed suction-cup mount. Keep the first path to the body and drop the loop closure, as
            # PhysX also poses a spanning tree and enforces the remaining joint as a constraint.
            if edge.child in posed:
                continue
            child_world = np.linalg.inv(edge.local_1) @ edge.motion @ edge.local_0 @ parent_world
            posed[edge.child] = child_world
            queue.append((edge.child, child_world))
    return posed


def _joint_body_paths(joint_prim: Usd.Prim) -> tuple[str | None, str | None]:
    """Return the (body0, body1) prim paths a joint connects, using None for a world attachment."""
    joint = UsdPhysics.Joint(joint_prim)
    body_paths: list[str | None] = []
    for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
        targets = relationship.GetTargets() if relationship else []
        body_paths.append(targets[0].pathString if targets else None)
    return body_paths[0], body_paths[1]


def _joint_local_frame(joint_prim: Usd.Prim, index: int) -> np.ndarray:
    """Return the joint frame relative to the joint's body at index (0 or 1)."""
    joint = UsdPhysics.Joint(joint_prim)
    position_attr = joint.GetLocalPos0Attr() if index == 0 else joint.GetLocalPos1Attr()
    rotation_attr = joint.GetLocalRot0Attr() if index == 0 else joint.GetLocalRot1Attr()
    position = position_attr.Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    rotation = rotation_attr.Get() or Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    quaternion = Gf.Quatd(float(rotation.GetReal()), Gf.Vec3d(*rotation.GetImaginary()))
    matrix = Gf.Matrix4d()
    matrix.SetTransform(Gf.Rotation(quaternion), Gf.Vec3d(*position))
    return np.array(matrix, dtype=np.float64)


def _joint_motion(joint_prim: Usd.Prim, value: float) -> np.ndarray:
    """Return the motion a movable joint applies at value, in its own joint frame."""
    axis_token = (
        UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get()
        if joint_prim.IsA(UsdPhysics.RevoluteJoint)
        else UsdPhysics.PrismaticJoint(joint_prim).GetAxisAttr().Get()
    )
    axis = _AXIS_VECTORS.get(axis_token or "X")
    assert axis is not None, f"Joint {joint_prim.GetPath()} has unsupported axis '{axis_token}'."

    matrix = Gf.Matrix4d()
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        matrix.SetRotate(Gf.Rotation(axis, np.degrees(value)))
    else:
        matrix.SetTranslate(axis * value)
    return np.array(matrix, dtype=np.float64)
