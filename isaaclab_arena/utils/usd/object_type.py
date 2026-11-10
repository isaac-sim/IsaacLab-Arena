# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate USD object types without loading simulation dependencies at import time."""

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.usd.variants import apply_usd_variant_selections


def validate_usd_object_type(
    usd_path: str,
    expected_type: ObjectType,
    variants: dict[str, str] | None = None,
) -> None:
    """Validate that a USD can be used as exactly one object of the expected type.

    Args:
        usd_path: Local or remote USD path to inspect.
        expected_type: Object type required by the consumer.
        variants: Variant selections to apply during inspection.
    """
    # Search backends are imported before SimulationApp starts. Load USD only when validating.
    from isaaclab.utils.assets import retrieve_file_path
    from pxr import Usd, UsdPhysics

    try:
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset", "Xform")
        root.GetReferences().AddReference(retrieve_file_path(usd_path))
        stage.SetDefaultPrim(root)
        if variants:
            apply_usd_variant_selections(stage, variants)
        if stage.GetCompositionErrors():
            raise ValueError("it has unresolved composition errors")
    except Exception as exc:
        raise ValueError(f"its USD could not be read: {exc}") from exc

    if expected_type == ObjectType.RIGID:
        bodies = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                raise ValueError("it has an articulation root")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(UsdPhysics.RigidBodyAPI(prim))
        if not bodies:
            raise ValueError("it has no rigid body")
        if len(bodies) != 1:
            raise ValueError(f"it has {len(bodies)} rigid bodies")
        if not bodies[0].GetRigidBodyEnabledAttr().Get():
            raise ValueError("its rigid body is disabled")
        return

    from isaaclab_arena.assets.object_utils import detect_object_type

    detected_type = detect_object_type(stage=stage)
    if detected_type != expected_type:
        raise ValueError(f"expected {expected_type.value}, found {detected_type.value}")
