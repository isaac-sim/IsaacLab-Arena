# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal cabinet adapter for reproducing scaled-articulation placement."""

from pathlib import Path

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.simready_central_asset_loader import _download_central_asset_package_to_cache
from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject


def _source_stage(url: str):
    """Reference the pinned SimReady cabinet into an editable wrapper stage."""
    from pxr import Usd, UsdGeom, UsdPhysics

    source_path = _download_central_asset_package_to_cache(url)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    source = stage.DefinePrim("/Asset/Cabinet")
    source.GetReferences().AddReference(source_path)
    for key, value in {"Physics": "physics", "PhysX": "Enabled"}.items():
        variants = source.GetVariantSets().GetVariantSet(key)
        if variants and value in variants.GetVariantNames():
            variants.SetVariantSelection(value)
    for prim in Usd.PrimRange(source):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        if prim.IsInstance():
            prim.SetInstanceable(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    return stage, source


def _save(stage, filename: str) -> str:
    cache = Path.home() / ".cache/isaaclab_arena/scaled_articulation_clutter_repro"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    stage.GetRootLayer().Export(str(path))
    return str(path)


@register_asset
class CleanupCabinet(SimReadyUsdObject, Openable):
    """Fixed-base SimReady cabinet with scale authored inside its wrapper USD."""

    registry_name = "cleanup_cabinet"
    tags = ["object", "sim-ready"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        usd_path: str,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_joint_positions: dict[str, float] | None = None,
        **kwargs,
    ):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        assert scale[0] == scale[1] == scale[2], "cleanup_cabinet requires a uniform scale"
        stage, source = _source_stage(usd_path)
        scale_factor = scale[0]
        UsdGeom.Xformable(source).AddScaleOp().Set(Gf.Vec3f(scale_factor))
        for prim in Usd.PrimRange(source):
            if prim.IsA(UsdPhysics.PrismaticJoint):
                joint = UsdPhysics.PrismaticJoint(prim)
                joint.GetLowerLimitAttr().Set(joint.GetLowerLimitAttr().Get() * scale_factor)
                joint.GetUpperLimitAttr().Set(joint.GetUpperLimitAttr().Get() * scale_factor)
                if initial_joint_positions is not None and prim.GetName() in initial_joint_positions:
                    drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
                    drive.CreateTargetPositionAttr(initial_joint_positions[prim.GetName()])
                    drive.CreateStiffnessAttr(300.0)
                    drive.CreateDampingAttr(50.0)
                    drive.CreateMaxForceAttr(100.0)
        UsdPhysics.ArticulationRootAPI.Apply(source)
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/Cabinet/FixedBase")
        fixed_joint.CreateBody1Rel().SetTargets(["/Asset/Cabinet/Geometry/body_01_obj_00"])
        super().__init__(
            usd_path=_save(stage, "cabinet.usda"),
            openable_joint_name="joint_part_01_joint_00",
            **kwargs,
        )
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = initial_joint_positions
