# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find the rigid bodies and joints in a USD asset.

Counting the rigid bodies is not enough to tell what an asset is. A bottle whose cap is held on
by a fixed joint and a cabinet whose doors swing on hinges both have several bodies, but only the
cabinet can actually move. Looking at the joints tells the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from pxr import Usd, UsdPhysics


@dataclass(frozen=True)
class JointInfo:
    """One enabled joint and the rigid bodies it connects."""

    path: str
    """Prim path of the joint."""

    type_name: str
    """USD type name, for example ``PhysicsRevoluteJoint``."""

    is_fixed: bool
    """True if the joint holds its bodies still relative to each other."""

    body_paths: tuple[str, ...]
    """Paths of the bodies the joint connects. A joint attached to the world has only one."""


@dataclass(frozen=True)
class PhysicsStructure:
    """The rigid bodies in a stage, grouped by which ones are held together."""

    rigid_body_paths: tuple[str, ...]
    """Every prim with a RigidBodyAPI, in the order they were found."""

    articulation_root_paths: tuple[str, ...]
    """Every prim with an ArticulationRootAPI."""

    joints: tuple[JointInfo, ...]
    """Enabled joints. Disabled joints hold nothing together, so they are skipped."""

    body_groups: tuple[tuple[str, ...], ...]
    """Bodies grouped by fixed joints. Each group moves as one piece."""

    @property
    def moving_joints(self) -> tuple[JointInfo, ...]:
        """The joints that still allow movement."""
        return tuple(joint for joint in self.joints if not joint.is_fixed)

    @property
    def is_single_rigid_body(self) -> bool:
        """True if the whole asset moves as one solid piece."""
        return len(self.body_groups) == 1 and not self.moving_joints


def _is_joint_enabled(prim: Usd.Prim) -> bool:
    """Read ``physics:jointEnabled``, which is true when nobody set it."""
    attribute = UsdPhysics.Joint(prim).GetJointEnabledAttr()
    if not attribute:
        return True
    value = attribute.Get()
    return True if value is None else bool(value)


def _find_body_for_prim(prim_path: str, rigid_body_paths: frozenset[str]) -> str | None:
    """Return the closest rigid body at or above prim_path.

    Joints usually point at a mesh inside a body rather than at the body itself, so we walk up
    the path until we hit a body.
    """
    path = prim_path
    while path not in ("", "/"):
        if path in rigid_body_paths:
            return path
        path = path.rsplit("/", 1)[0]
    return None


def _get_joint_bodies(prim: Usd.Prim, rigid_body_paths: frozenset[str]) -> tuple[str, ...]:
    """Return the rigid bodies a joint connects."""
    joint = UsdPhysics.Joint(prim)
    body_paths: list[str] = []
    for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
        for target in relationship.GetTargets():
            body_path = _find_body_for_prim(str(target), rigid_body_paths)
            if body_path is not None and body_path not in body_paths:
                body_paths.append(body_path)
    return tuple(body_paths)


def _group_bodies_by_fixed_joints(
    rigid_body_paths: list[str], joints: tuple[JointInfo, ...]
) -> tuple[tuple[str, ...], ...]:
    """Group bodies that fixed joints hold together."""
    parent = {path: path for path in rigid_body_paths}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for joint in joints:
        if joint.is_fixed:
            # A joint attached to the world has fewer than two bodies, so this loop does nothing.
            for body_path in joint.body_paths[1:]:
                parent[find(body_path)] = find(joint.body_paths[0])

    groups: dict[str, list[str]] = {}
    for path in rigid_body_paths:
        groups.setdefault(find(path), []).append(path)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def get_physics_structure(stage: Usd.Stage) -> PhysicsStructure:
    """Find the rigid bodies, joints, and body groups in an open stage.

    Select any physics variant on the stage first. See ``apply_usd_variant_selections``.

    Args:
        stage: Open USD stage to look at.

    Returns:
        The rigid bodies and joints found in the stage.
    """
    rigid_body_paths: list[str] = []
    articulation_root_paths: list[str] = []
    joint_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_paths.append(path)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_root_paths.append(path)
        if prim.IsA(UsdPhysics.Joint) and _is_joint_enabled(prim):
            joint_prims.append(prim)

    body_path_set = frozenset(rigid_body_paths)
    joints = tuple(
        JointInfo(
            path=str(prim.GetPath()),
            type_name=str(prim.GetTypeName()),
            is_fixed=bool(prim.IsA(UsdPhysics.FixedJoint)),
            body_paths=_get_joint_bodies(prim, body_path_set),
        )
        for prim in joint_prims
    )
    return PhysicsStructure(
        rigid_body_paths=tuple(rigid_body_paths),
        articulation_root_paths=tuple(articulation_root_paths),
        joints=joints,
        body_groups=_group_bodies_by_fixed_joints(rigid_body_paths, joints),
    )
