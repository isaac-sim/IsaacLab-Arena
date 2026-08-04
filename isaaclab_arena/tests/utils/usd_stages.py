# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build small in-memory USD stages shaped like SimReady props."""

from __future__ import annotations

from typing import Any


def new_stage() -> Any:
    """Create an in-memory stage with a ``/Root`` default prim."""
    # This helper is called from unit tests, which may run before SimulationApp starts.
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def add_body(stage: Any, name: str) -> str:
    """Add a rigid body shaped the way SimReady authors them: an Xform with a mesh under it.

    Args:
        stage: Stage to add to.
        name: Part name, for example ``body_01``.

    Returns:
        Prim path of the body with the RigidBodyAPI.
    """
    # This helper is called from unit tests, which may run before SimulationApp starts.
    from pxr import UsdGeom, UsdPhysics

    body_path = f"/Root/Geometry/{name}_obj_00"
    body = UsdGeom.Xform.Define(stage, body_path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"{body_path}/{name}_mesh_00")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return body_path
