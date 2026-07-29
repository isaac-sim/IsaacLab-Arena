# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build small in-memory USD stages shaped like SimReady props.

pxr is imported inside each function, so importing this module is safe during pytest collection,
before Isaac Sim has started.
"""

from __future__ import annotations

from typing import Any


def new_stage() -> Any:
    """Create an in-memory stage with a ``/Root`` default prim."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def add_body(stage: Any, name: str) -> str:
    """Add a rigid body shaped like SimReady authors them: an Xform with a mesh under it.

    Args:
        stage: Stage to add to.
        name: Part name, for example ``body_01``.

    Returns:
        Prim path of the body with the RigidBodyAPI.
    """
    from pxr import UsdGeom, UsdPhysics

    body_path = f"/Root/Geometry/{name}_obj_00"
    body = UsdGeom.Xform.Define(stage, body_path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdGeom.Mesh.Define(stage, f"{body_path}/{name}_mesh_00")
    return body_path


def add_joint(
    stage: Any,
    name: str,
    kind: str,
    body0: str | None,
    body1: str | None,
    enabled: bool = True,
) -> None:
    """Join two bodies, pointing at their meshes the way SimReady does.

    Args:
        stage: Stage to add to.
        name: Joint prim name.
        kind: Either ``fixed`` or ``revolute``.
        body0: Prim path of the first body, or None to attach to the world.
        body1: Prim path of the second body, or None to attach to the world.
        enabled: Value for ``physics:jointEnabled``.
    """
    from pxr import UsdPhysics

    schema = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint}[kind]
    joint = schema.Define(stage, f"/Root/Joints/{name}")
    for relationship, body in ((joint.CreateBody0Rel(), body0), (joint.CreateBody1Rel(), body1)):
        if body is not None:
            mesh_name = body.rsplit("/", 1)[-1].replace("_obj_00", "_mesh_00")
            relationship.SetTargets([f"{body}/{mesh_name}"])
    joint.CreateJointEnabledAttr(enabled)


def welded_bodies_stage() -> Any:
    """Two bodies joined by a fixed joint, like a spray bottle and its cap."""
    stage = new_stage()
    body = add_body(stage, "body_01")
    cup = add_body(stage, "cup_01")
    add_joint(stage, "joint_cup_01", "fixed", body, cup)
    return stage


def hinged_bodies_stage() -> Any:
    """A frame with a swinging door and a handle fixed to that door, like a cabinet."""
    stage = new_stage()
    carcass = add_body(stage, "body_01")
    door = add_body(stage, "part_01")
    handle = add_body(stage, "part_02")
    add_joint(stage, "joint_door", "revolute", carcass, door)
    add_joint(stage, "joint_handle", "fixed", door, handle)
    return stage
