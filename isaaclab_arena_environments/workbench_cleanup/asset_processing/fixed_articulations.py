# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create fixed-base articulations from partitioned frame meshes and existing jointed bodies.

The operation selects a partitioned frame collider and prismatic or revolute
joints that share one source frame. Processing creates one articulation under
``Articulations``, copies the frame collider and moving rigid bodies into it,
rewires the copied joints, and fixes the new frame rigid body to the original
source frame. The selected source mesh, bodies, and joints are then deactivated.

For example, this input:

.. code-block:: text

   /Asset
     /Geometry/Frame                 (RigidBody)
       /PanelPartition               (Mesh, Collision)
       /MovingPanel                  (RigidBody)
     /Joints/PanelSlide              (PrismaticJoint)

   {
       "operation": "fixed_articulation",
       "name": "PanelAssembly",
       "frame_mesh": "Geometry/Frame/PanelPartition",
       "joints": ["Joints/PanelSlide"]
   }

produces:

.. code-block:: text

   /Asset
     /Geometry/Frame                 (original RigidBody removed)
       /PanelPartition               (inactive)
       /MovingPanel                  (inactive)
     /Joints/PanelSlide              (inactive)
     /Articulations/PanelAssembly     (ArticulationRoot)
       /Frame                        (RigidBody)
         /Geometry                   (copied PanelPartition)
       /MovingPanel                  (copied RigidBody)
       /Joints/PanelSlide             (rewired copied joint)
       /Joints/FixedBase              (FixedJoint to original frame)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Gf, Sdf, Usd


def create_fixed_articulation(
    stage: Usd.Stage,
    name: str,
    frame_mesh: str,
    joints: list[str],
) -> None:
    """Create one fixed-base articulation from selected source joints.

    Args:
        stage: USD stage whose default prim contains the source bodies.
        name: Name of the articulation created under ``Articulations``.
        frame_mesh: Asset-relative path to the partitioned frame collider.
        joints: Asset-relative paths to movable joints sharing one source frame.

    Returns:
        None.
    """
    from pxr import Sdf, Tf, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    assert root, "Expected an asset default prim."
    root_path = root.GetPath()

    def prim_at(path: str | Sdf.Path) -> Usd.Prim:
        """Return a validated prim beneath the asset root."""
        path = Sdf.Path(path)
        if not path.IsAbsolutePath():
            path = root_path.AppendPath(path)
        assert path.IsPrimPath() and path.HasPrefix(root_path), "Expected an asset-relative prim path."
        prim = stage.GetPrimAtPath(path)
        assert prim, f"Missing prim: {path}."
        return prim

    assert isinstance(name, str) and Tf.IsValidIdentifier(name), f"Invalid articulation name: {name}."
    destination = root_path.AppendPath(f"Articulations/{name}")
    assert not stage.GetPrimAtPath(destination), f"Destination exists: {destination}."
    mesh = prim_at(frame_mesh)
    assert mesh.IsA(UsdGeom.Mesh) and mesh.HasAPI(UsdPhysics.CollisionAPI), "Frame must be a mesh collider."
    assert isinstance(joints, list) and joints, "Expected existing movable joints."

    selected_joints = set()
    selected_bodies = set()
    joint_specs = []
    base_path = None
    for path in joints:
        prim = prim_at(path)
        assert prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(
            UsdPhysics.RevoluteJoint
        ), "Only prismatic and revolute joints are supported."
        assert prim.GetPath() not in selected_joints, "Joint selected more than once."
        selected_joints.add(prim.GetPath())
        joint = UsdPhysics.Joint(prim)
        targets_0, targets_1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        assert len(targets_0) == len(targets_1) == 1, "Both joint endpoints must be explicit."
        if base_path is None:
            base_path = targets_0[0]
        else:
            assert targets_0[0] == base_path, "All joints must attach directly to the same source frame."
        owner = _rigid_owner(stage.GetPrimAtPath(targets_1[0]))
        assert owner.GetPath() not in selected_bodies, "Moving bodies must have exactly one selected joint."
        selected_bodies.add(owner.GetPath())
        joint_specs.append((prim.GetPath(), owner.GetPath(), targets_1[0]))

    assert base_path is not None
    source_owner = _rigid_owner(stage.GetPrimAtPath(base_path))
    assert mesh.GetPath() != base_path and mesh.GetPath().HasPrefix(
        base_path
    ), "Selected frame mesh must be a partition beneath the source joint frame."
    assert source_owner.GetPath() not in selected_bodies, "Articulated chains are unsupported."
    affected_bodies = selected_bodies | {source_owner.GetPath()}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetPath() in selected_joints:
            continue
        joint = UsdPhysics.Joint(prim)
        for target in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets():
            for owner in affected_bodies:
                assert not target.HasPrefix(owner), f"Unselected joint {prim.GetPath()} uses a moved body."

    for path in {mesh.GetPath()} | affected_bodies:
        prim = stage.GetPrimAtPath(path)
        ancestors = []
        while prim and not prim.IsPseudoRoot():
            ancestors.append(prim.GetPath())
            prim = prim.GetParent()
        for ancestor in reversed(ancestors):
            prim = stage.GetPrimAtPath(ancestor)
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
    cache = UsdGeom.XformCache()
    T_W_R_inverse = cache.GetLocalToWorldTransform(root).GetInverse()

    def pose_R(path: Sdf.Path) -> Gf.Matrix4d:
        """Return a prim transform in the default-prim frame."""
        prim = stage.GetPrimAtPath(path)
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            xform = UsdGeom.Xformable(ancestor)
            assert not (xform and xform.TransformMightBeTimeVarying()), "Animated body transforms are unsupported."
            ancestor = ancestor.GetParent()
        T_R_S = cache.GetLocalToWorldTransform(prim) * T_W_R_inverse
        linear = np.asarray(T_R_S)[:3, :3]
        assert np.allclose(linear @ linear.T, np.eye(3), atol=1e-5), "Scaled or sheared body frames are unsupported."
        assert np.linalg.det(linear) > 0, "Reflected body frames are unsupported."
        return T_R_S

    snapshot = stage.Flatten()
    _copy_flattened_prototypes(stage, snapshot)
    T_R_B = pose_R(base_path)
    articulation = UsdGeom.Xform.Define(stage, destination)
    articulation.AddTransformOp().Set(T_R_B)
    UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
    frame_path = destination.AppendChild("Frame")
    frame = UsdGeom.Xform.Define(stage, frame_path)
    UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
    _copy_at_asset_pose(
        stage,
        snapshot,
        mesh.GetPath(),
        frame_path.AppendChild("Geometry"),
        pose_R(mesh.GetPath()),
        T_R_B,
    )
    for joint_path, owner_path, endpoint_path in joint_specs:
        new_body_path = destination.AppendChild(owner_path.name)
        assert not stage.GetPrimAtPath(new_body_path), f"Duplicate body name: {new_body_path}."
        _copy_at_asset_pose(stage, snapshot, owner_path, new_body_path, pose_R(owner_path), T_R_B)
        new_joint_path = destination.AppendPath(f"Joints/{joint_path.name}")
        stage.OverridePrim(new_joint_path)
        assert Sdf.CopySpec(snapshot, joint_path, stage.GetEditTarget().GetLayer(), new_joint_path)
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(new_joint_path))
        joint.GetBody0Rel().SetTargets([frame_path])
        joint.GetBody1Rel().SetTargets([endpoint_path.ReplacePrefix(owner_path, new_body_path)])
    fixed = UsdPhysics.FixedJoint.Define(stage, destination.AppendPath("Joints/FixedBase"))
    fixed.CreateBody0Rel().SetTargets([base_path])
    fixed.CreateBody1Rel().SetTargets([frame_path])

    for path in {mesh.GetPath()} | selected_bodies | selected_joints:
        stage.GetPrimAtPath(path).SetActive(False)
    prim = source_owner
    for schema in prim.GetAppliedSchemas():
        if "RigidBody" in schema or schema == "PhysicsMassAPI":
            prim.RemoveAppliedSchema(schema)
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        prim = prim.GetParent()


def _rigid_owner(prim: Usd.Prim) -> Usd.Prim:
    """Return the closest rigid-body ancestor of a joint endpoint."""
    from pxr import UsdPhysics

    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        prim = prim.GetParent()
    raise AssertionError("Joint endpoint has no rigid-body owner.")


def _copy_at_asset_pose(
    stage: Usd.Stage,
    snapshot: Sdf.Layer,
    source: Sdf.Path,
    destination: Sdf.Path,
    T_R_S: Gf.Matrix4d,
    T_R_P: Gf.Matrix4d,
) -> None:
    """Copy a prim while preserving its transform in the asset frame."""
    from pxr import Sdf, UsdGeom

    stage.OverridePrim(destination)
    assert Sdf.CopySpec(snapshot, source, stage.GetEditTarget().GetLayer(), destination)
    prim = stage.GetPrimAtPath(destination)
    prim.SetInstanceable(False)
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:") or attr.GetName() == "xformOpOrder":
            prim.RemoveProperty(attr.GetName())
    UsdGeom.Xformable(prim).AddTransformOp().Set(T_R_S * T_R_P.GetInverse())


def _copy_flattened_prototypes(stage: Usd.Stage, snapshot: Sdf.Layer) -> None:
    """Copy flattened prototype specs needed by copied instanceable prims."""
    from pxr import Sdf

    destination = stage.GetEditTarget().GetLayer()
    for prim_spec in snapshot.rootPrims:
        if prim_spec.name.startswith("Flattened_Prototype") and not destination.GetPrimAtPath(prim_spec.path):
            assert Sdf.CopySpec(snapshot, prim_spec.path, destination, prim_spec.path)
