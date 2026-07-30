# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build small in-memory USD stages shaped like SimReady props."""

from __future__ import annotations

from typing import Any


def new_stage() -> Any:
    """Create an in-memory stage with a ``/Root`` default prim."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def add_body(
    stage: Any,
    name: str,
    mass: float | None = None,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    diagonal_inertia: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] | None = None,
) -> str:
    """Add a rigid body shaped like SimReady authors them: an Xform with a mesh under it.

    Args:
        stage: Stage to add to.
        name: Part name, for example ``body_01``.
        mass: Mass to write on the body. No MassAPI is applied when this is None.
        center_of_mass: Centre of mass, in the body's own frame.
        diagonal_inertia: Moments of inertia about the body's own axes.
        translation: Where to put the body, if not at the origin.

    Returns:
        Prim path of the body with the RigidBodyAPI.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    body_path = f"/Root/Geometry/{name}_obj_00"
    body = UsdGeom.Xform.Define(stage, body_path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    if translation is not None:
        body.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if mass is not None:
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr().Set(mass)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia))
    mesh = UsdGeom.Mesh.Define(stage, f"{body_path}/{name}_mesh_00")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
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


def fixed_joint_bodies_stage() -> Any:
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
