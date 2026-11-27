# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a simulation workcell from the same geometry used by the scene exporter."""

import fnmatch
import tempfile
from functools import lru_cache
from pathlib import Path

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from isaaclab_arena_environments.spatial_memory.workcells.appearance import add_library_visuals
from isaaclab_arena_environments.spatial_memory.workcells.bodies import BODIES, SLIDE_BUTTON
from isaaclab_arena_environments.spatial_memory.workcells.catalog import build_stage
from isaaclab_arena_environments.spatial_memory.workcells.geometry import Workcell
from isaaclab_arena_environments.spatial_memory.workcells.layouts import (
    FORBIDDEN_CONTACT,
    LAYOUTS,
    PILOT_STARTS,
    REQUIRED_SUPPORT,
)


def _add_layout(stage, name, layout):
    spec = LAYOUTS[name]
    cell = Workcell(stage)
    for label, center, size in spec.guards:
        cell.box(label, center, size, "steel")
    if name == "stack_bracing":
        stage.GetPrimAtPath(cell.parts.AppendChild("RearFoot")).GetAttribute("xformOp:translate").Set(
            spec.centers[layout]
        )
    else:
        cell.box(spec.feature, spec.centers[layout], spec.size, "limit")
    if name == "panel_catches":
        stage.RemovePrim(cell.parts.AppendChild("Catch0"))
        stage.GetPrimAtPath(cell.parts.AppendChild("CatchMount")).GetAttribute("xformOp:translate").Set(
            (-0.25, 0.093, 1.06)
        )
        stage.GetPrimAtPath(cell.parts.AppendChild("Catch1")).GetAttribute("xformOp:translate").Set((0.25, 0.035, 1.06))
    if name == "cabinet_reach":
        stage.GetPrimAtPath(cell.parts.AppendChild("ResetPad")).GetAttribute("xformOp:translate").Set(
            (0.14, 0.247, 1.06)
        )
    if name in {"shelf_bay", "partition_delivery"}:
        load = stage.GetPrimAtPath(cell.parts.AppendChild("load"))
        load.GetAttribute("xformOp:scale").Set((0.20, 0.3588, 0.40))
    if name in PILOT_STARTS:
        stage.GetPrimAtPath(cell.parts.AppendChild("load")).GetAttribute("xformOp:translate").Set(PILOT_STARTS[name][2])
    if name == "seam_inspection":
        last = UsdGeom.Cylinder(stage.GetPrimAtPath(cell.parts.AppendChild("Weld3")))
        transform = last.ComputeLocalToWorldTransform(0)
        start = transform.Transform(Gf.Vec3d(0, 0, -last.GetHeightAttr().Get() / 2))
        endpoint = Gf.Vec3d(*spec.centers[layout])
        stage.RemovePrim(last.GetPath())
        cell.bar("Weld3", start, endpoint, 0.012, "target")
    if name == "display_cover":
        cell.group("PeelHinge", (0, 0.005, 0.939))
    if name == "worktop_clamp":
        # Start the lower jaw open, below the underside bearing pad.
        stage.GetPrimAtPath(cell.parts.AppendChild("LowerJaw")).GetAttribute("xformOp:translate").Set(
            (0.0, -0.45, 0.74)
        )
    if name == "rail_support":
        stage.GetPrimAtPath(cell.parts.AppendChild("Crate")).GetAttribute("xformOp:translate").Set((0, -0.57, 1.13))


def _attach_joint(stage, spec, actor, anchor, pivot, anchor_origin):
    slide = spec.kind == "slide"
    joint_type = UsdPhysics.PrismaticJoint if slide else UsdPhysics.RevoluteJoint
    joint = joint_type.Define(stage, actor.GetPath().GetParentPath().AppendChild(actor.GetName() + "Joint"))
    joint.CreateBody0Rel().SetTargets([anchor.GetPath()])
    joint.CreateBody1Rel().SetTargets([actor.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(pivot - anchor_origin))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateAxisAttr(spec.axis)
    if spec.limits is not None:
        joint.CreateLowerLimitAttr(spec.limits[0])
        joint.CreateUpperLimitAttr(spec.limits[1])
    if spec == SLIDE_BUTTON:
        spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        spring.CreateStiffnessAttr(65.0)
        spring.CreateDampingAttr(2.0)
        spring.CreateTargetPositionAttr(0.0)
    if spec.parent == "clamp":
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr(1.0)


def _store_motion_samples(actor, pivot_prim, cache):
    """Keep the source keyframes in seconds for per-environment kinematic playback."""
    frames = UsdGeom.Xformable(pivot_prim).GetTimeSamples()
    poses = []
    for frame in frames:
        transform = UsdGeom.Xformable(pivot_prim).GetLocalTransformation(frame)
        initial = cache.GetLocalToWorldTransform(pivot_prim).ExtractRotation()
        yaw = (initial.GetInverse() * transform.ExtractRotation()).Decompose(
            Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis()
        )[0]
        poses.append(Gf.Vec4d(*transform.ExtractTranslation(), yaw))

    actor.GetPrim().SetCustomDataByKey(
        "motionTimes", Vt.DoubleArray([f / pivot_prim.GetStage().GetTimeCodesPerSecond() for f in frames])
    )
    actor.GetPrim().SetCustomDataByKey("motionPoses", Vt.Vec4dArray(poses))


def _add_pickup_rest(stage, fixtures, actor):
    """Support otherwise airborne workpieces until the robot picks them up."""
    bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(actor.GetPrim()).ComputeAlignedRange()
    center = bounds.GetMidpoint()
    bottom = bounds.GetMin()[2]
    stand = UsdGeom.Cube.Define(stage, fixtures.GetPath().AppendChild(actor.GetPrim().GetName() + "Rest"))
    stand.CreateSizeAttr(1)
    stand.AddTranslateOp().Set((center[0], center[1], bottom / 2))
    stand.AddScaleOp().Set((0.085, 0.085, bottom))
    stand.CreateDisplayColorAttr([(0.32, 0.40, 0.42)])
    UsdPhysics.CollisionAPI.Apply(stand.GetPrim())


def _copy_shape(prim, stage, path, T_parent_shape):
    """Keep authored shape dimensions, material and collision schema when reparenting a shape."""
    Sdf.CopySpec(prim.GetStage().GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), path)
    UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(T_parent_shape)


def physical_stage(name, layout):
    """Return the fixed fixtures and resettable rigid bodies for one guide layout."""
    assert layout in (0, 1)
    source = build_stage(name, 3847)
    _add_layout(source, name, layout)
    cell = source.GetPrimAtPath("/World/Workcell")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Workcell").GetPrim()
    stage.SetDefaultPrim(root)
    fixtures = UsdGeom.Xform.Define(stage, "/Workcell/Fixtures")
    cache = UsdGeom.XformCache(Usd.TimeCode(0))
    parts = {str(p.GetPath().MakeRelativePath(cell.GetPath())): p for p in Usd.PrimRange(cell) if p.IsA(UsdGeom.Gprim)}
    pivots = {}
    for spec in BODIES[name]:
        pivot_prim = source.GetPrimAtPath(cell.GetPath().AppendPath(spec.pivot))
        pivot = cache.GetLocalToWorldTransform(pivot_prim).ExtractTranslation()
        pivots[spec.name] = pivot
        actor = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("Body_" + spec.name))
        actor.AddTranslateOp().Set(pivot)
        rigid = UsdPhysics.RigidBodyAPI.Apply(actor.GetPrim())
        rigid.CreateKinematicEnabledAttr(spec.name == "mover")
        if name == "moving_carriage" and spec.name == "part":
            PhysxSchema.PhysxContactReportAPI.Apply(actor.GetPrim()).CreateThresholdAttr(0.0)
        if spec.name == "mover":
            _store_motion_samples(actor, pivot_prim, cache)
        UsdPhysics.MassAPI.Apply(actor.GetPrim()).CreateMassAttr(spec.mass_kg)
        matches = [key for key in parts if any(fnmatch.fnmatchcase(key, pat) for pat in spec.shapes.split())]
        assert matches, f"No collision geometry matched {name}/{spec.name}"
        for key in matches:
            prim = parts.pop(key)
            path = actor.GetPath().AppendChild(prim.GetName())
            _copy_shape(prim, stage, path, cache.GetLocalToWorldTransform(prim) * Gf.Matrix4d().SetTranslate(-pivot))
        if spec.mechanism:
            if spec.mechanism.parent is not None:
                anchor = stage.GetPrimAtPath("/Workcell/Body_" + spec.mechanism.parent)
                anchor_origin = pivots[spec.mechanism.parent]
            else:
                anchor = UsdGeom.Xform.Define(stage, "/Workcell/Anchor").GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(anchor).CreateKinematicEnabledAttr(True)
                anchor_origin = Gf.Vec3d(0)
            _attach_joint(stage, spec.mechanism, actor.GetPrim(), anchor, pivot, anchor_origin)
        if spec.stand:
            _add_pickup_rest(stage, fixtures, actor)
    instrumented = [(surface, "Support_" + surface) for surface, _ in REQUIRED_SUPPORT.get(name, ())]
    if name in FORBIDDEN_CONTACT:
        instrumented.append((LAYOUTS[name].feature, "Guard"))
    for key, label in instrumented:
        prim = parts.pop(key)
        sensor_body = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild(label))
        t_E_sensor = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        sensor_body.AddTranslateOp().Set(t_E_sensor)
        UsdPhysics.RigidBodyAPI.Apply(sensor_body.GetPrim()).CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(sensor_body.GetPrim()).CreateThresholdAttr(0.0)
        path = sensor_body.GetPath().AppendChild(key)
        _copy_shape(prim, stage, path, cache.GetLocalToWorldTransform(prim) * Gf.Matrix4d().SetTranslate(-t_E_sensor))
    for key, prim in parts.items():
        path = fixtures.GetPath().AppendChild(key.replace("/", "_"))
        _copy_shape(prim, stage, path, cache.GetLocalToWorldTransform(prim))
    return stage


@lru_cache(maxsize=60)
def workcell_usd(name, layout):
    """Keep one compiled USD per requested workcell and layout for the lifetime of this process."""
    directory = tempfile.TemporaryDirectory(prefix="arena-g1-workcell-")
    path = Path(directory.name) / f"{name}.usdc"
    stage = physical_stage(name, layout)
    add_library_visuals(stage, name)
    if not stage.GetRootLayer().Export(str(path)):
        raise OSError(f"Could not write workcell {path}")
    return directory, str(path)
