# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create fixed cabinet articulations from partitioned frames and existing joints."""

from __future__ import annotations

import numpy as np
from typing import Any


def create_fixed_articulations(stage, groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Move selected frame meshes and jointed bodies into separate fixed articulations."""
    from pxr import Sdf, Tf, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    assert root and isinstance(groups, dict) and groups, "Expected named fixed articulations."
    root_path = root.GetPath()

    def prim_at(path):
        path = Sdf.Path(path)
        if not path.IsAbsolutePath():
            path = root_path.AppendPath(path)
        assert path.IsPrimPath() and path.HasPrefix(root_path), "Expected an asset-relative prim path."
        prim = stage.GetPrimAtPath(path)
        assert prim, f"Missing prim: {path}."
        return prim

    selected_joints = set()
    selected_meshes = set()
    selected_bodies = set()
    source_owners = set()
    plans = []
    for name, config in groups.items():
        assert isinstance(name, str) and Tf.IsValidIdentifier(name), f"Invalid articulation name: {name}."
        assert isinstance(config, dict) and set(config) == {
            "frame_mesh",
            "joints",
        }, "Each fixed articulation requires frame_mesh and joints."
        destination = root_path.AppendPath(f"Articulations/{name}")
        assert not stage.GetPrimAtPath(destination), f"Destination exists: {destination}."
        mesh = prim_at(config["frame_mesh"])
        assert mesh.IsA(UsdGeom.Mesh) and mesh.HasAPI(UsdPhysics.CollisionAPI), "Frame must be a mesh collider."
        assert mesh.GetPath() not in selected_meshes, "Frame mesh selected more than once."
        selected_meshes.add(mesh.GetPath())
        assert isinstance(config["joints"], list) and config["joints"], "Expected existing movable joints."
        joints = []
        base_path = None
        for path in config["joints"]:
            prim = prim_at(path)
            assert prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(
                UsdPhysics.RevoluteJoint
            ), "Only prismatic and revolute drawer/door joints are supported."
            assert prim.GetPath() not in selected_joints, "Joint selected more than once."
            selected_joints.add(prim.GetPath())
            joint = UsdPhysics.Joint(prim)
            targets_0, targets_1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            assert len(targets_0) == len(targets_1) == 1, "Both joint endpoints must be explicit."
            if base_path is None:
                base_path = targets_0[0]
            assert targets_0[0] == base_path, "All joints must attach directly to the same source frame."
            owner = _rigid_owner(stage.GetPrimAtPath(targets_1[0]))
            assert owner.GetPath() not in selected_bodies, "Moving bodies must have exactly one selected joint."
            selected_bodies.add(owner.GetPath())
            joints.append((prim.GetPath(), owner.GetPath(), targets_1[0]))
        base = stage.GetPrimAtPath(base_path)
        source_owner = _rigid_owner(base)
        assert mesh.GetPath() != base_path and mesh.GetPath().HasPrefix(
            base_path
        ), "Selected frame mesh must be a partition beneath the source joint frame."
        assert source_owner.GetPath() not in selected_bodies, "Articulated chains are unsupported."
        source_owners.add(source_owner.GetPath())
        plans.append((destination, mesh.GetPath(), base_path, joints))

    assert not source_owners & selected_bodies, "A source frame cannot also be a moving body."
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetPath() in selected_joints:
            continue
        joint = UsdPhysics.Joint(prim)
        for target in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets():
            for owner in source_owners | selected_bodies:
                assert not target.HasPrefix(owner), f"Unselected joint {prim.GetPath()} uses a moved body."

    for path in selected_meshes | selected_bodies | source_owners:
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

    def pose_R(path):
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
    reports = []
    for destination, mesh_path, base_path, joints in plans:
        T_R_B = pose_R(base_path)
        articulation = UsdGeom.Xform.Define(stage, destination)
        articulation.AddTransformOp().Set(T_R_B)
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
        frame_path = destination.AppendChild("Frame")
        frame = UsdGeom.Xform.Define(stage, frame_path)
        UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
        _copy_at_asset_pose(stage, snapshot, mesh_path, frame_path.AppendChild("Geometry"), pose_R(mesh_path), T_R_B)
        body_paths = []
        joint_paths = []
        for joint_path, owner_path, endpoint_path in joints:
            new_body_path = destination.AppendChild(owner_path.name)
            assert not stage.GetPrimAtPath(new_body_path), f"Duplicate body name: {new_body_path}."
            _copy_at_asset_pose(stage, snapshot, owner_path, new_body_path, pose_R(owner_path), T_R_B)
            new_joint_path = destination.AppendPath(f"Joints/{joint_path.name}")
            stage.OverridePrim(new_joint_path)
            assert Sdf.CopySpec(snapshot, joint_path, stage.GetEditTarget().GetLayer(), new_joint_path)
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(new_joint_path))
            joint.GetBody0Rel().SetTargets([frame_path])
            joint.GetBody1Rel().SetTargets([endpoint_path.ReplacePrefix(owner_path, new_body_path)])
            body_paths.append(str(new_body_path))
            joint_paths.append(str(new_joint_path))
        fixed = UsdPhysics.FixedJoint.Define(stage, destination.AppendPath("Joints/FixedBase"))
        fixed.CreateBody0Rel().SetTargets([base_path])
        fixed.CreateBody1Rel().SetTargets([frame_path])
        reports.append({
            "articulation": str(destination),
            "base": str(frame_path),
            "frame_mesh": str(frame_path.AppendChild("Geometry")),
            "bodies": body_paths,
            "joints": joint_paths,
        })

    for path in selected_meshes | selected_bodies | selected_joints:
        stage.GetPrimAtPath(path).SetActive(False)
    for path in source_owners:
        prim = stage.GetPrimAtPath(path)
        for schema in prim.GetAppliedSchemas():
            if "RigidBody" in schema or schema == "PhysicsMassAPI":
                prim.RemoveAppliedSchema(schema)
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim = prim.GetParent()
    return reports


def _rigid_owner(prim):
    from pxr import UsdPhysics

    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        prim = prim.GetParent()
    raise AssertionError("Joint endpoint has no rigid-body owner.")


def _copy_at_asset_pose(stage, snapshot, source, destination, T_R_S, T_R_P):
    from pxr import Sdf, UsdGeom

    stage.OverridePrim(destination)
    assert Sdf.CopySpec(snapshot, source, stage.GetEditTarget().GetLayer(), destination)
    prim = stage.GetPrimAtPath(destination)
    prim.SetInstanceable(False)
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:") or attr.GetName() == "xformOpOrder":
            prim.RemoveProperty(attr.GetName())
    UsdGeom.Xformable(prim).AddTransformOp().Set(T_R_S * T_R_P.GetInverse())
    return prim


def _copy_flattened_prototypes(stage, snapshot) -> None:
    from pxr import Sdf

    destination = stage.GetEditTarget().GetLayer()
    for prim_spec in snapshot.rootPrims:
        if prim_spec.name.startswith("Flattened_Prototype") and not destination.GetPrimAtPath(prim_spec.path):
            assert Sdf.CopySpec(snapshot, prim_spec.path, destination, prim_spec.path)
