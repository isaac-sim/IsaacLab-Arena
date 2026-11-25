# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build reusable, metre-scale workcell geometry on an OpenUSD stage (Z up, front toward -Y)."""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

COLORS = {
    "frame": (0.24, 0.33, 0.38),
    "steel": (0.52, 0.65, 0.69),
    "wood": (0.61, 0.39, 0.22),
    "work": (0.95, 0.43, 0.12),
    "target": (0.08, 0.55, 0.49),
    "limit": (0.88, 0.64, 0.22),
    "floor": (0.36, 0.39, 0.40),
}


class Workcell:
    """Author physical scene components and two inspection cameras."""

    def __init__(self, stage):
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        self.parts = UsdGeom.Xform.Define(stage, "/World/Workcell").GetPath()

    def _child_path(self, parent, name):
        path = (parent or self.parts).AppendChild(name)
        assert not self.stage.GetPrimAtPath(path), f"Duplicate component {path}"
        return path

    def group(self, name, xyz=(0, 0, 0), rotation=(0, 0, 0), parent=None):
        path = self._child_path(parent, name)
        node = UsdGeom.Xform.Define(self.stage, path)
        node.AddTranslateOp().Set(xyz)
        node.AddRotateXYZOp().Set(rotation)
        return path

    def _shape(self, schema, xyz, color, rotation):
        schema.AddTranslateOp().Set(xyz)
        schema.AddRotateXYZOp().Set(rotation)
        schema.CreateDisplayColorAttr([COLORS[color] if isinstance(color, str) else color])
        UsdPhysics.CollisionAPI.Apply(schema.GetPrim())
        return schema.GetPath()

    def box(self, name, xyz, size, color="steel", rotation=(0, 0, 0), parent=None):
        path = self._child_path(parent, name)
        shape = UsdGeom.Cube.Define(self.stage, path)
        shape.CreateSizeAttr(1)
        self._shape(shape, xyz, color, rotation)
        shape.AddScaleOp().Set(size)
        return path

    def cylinder(self, name, xyz, radius, length, color="steel", rotation=(0, 0, 0), parent=None):
        path = self._child_path(parent, name)
        shape = UsdGeom.Cylinder.Define(self.stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(length)
        return self._shape(shape, xyz, color, rotation)

    def bar(self, name, start, end, radius, color="steel", parent=None):
        """Place a round bar between two endpoints in its parent's frame."""
        a, b = Gf.Vec3d(*start), Gf.Vec3d(*end)
        delta = b - a
        assert delta.GetLength() > 0, name
        path = self.cylinder(name, (a + b) / 2, radius, delta.GetLength(), color, parent=parent)
        axis = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()
        UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).AddOrientOp().Set(Gf.Quatf(axis))
        return path

    def bench(self, width=1.3, depth=0.85, height=0.82):
        self.box("BenchTop", (0, 0, height - 0.03), (width, depth, 0.06), "wood")
        for i, (x, y) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
            self.box(
                f"BenchLeg{i}",
                (x * (width / 2 - 0.05), y * (depth / 2 - 0.05), (height - 0.06) / 2),
                (0.055, 0.055, height - 0.06),
                "frame",
            )

    def tray(self, name, xyz, size, color="work", rotation=(0, 0, 0)):
        """Create an open tray with four rims; xyz is the center of the bottom plate."""
        w, d, h = size
        parent = self.group(name, xyz, rotation)
        thick = 0.018
        self.box("Base", (0, 0, 0), (w, d, thick), color, parent=parent)
        for i, sign in enumerate((-1, 1)):
            self.box(f"End{i}", (0, sign * (d - thick) / 2, h / 2), (w, thick, h), color, parent=parent)
            self.box(f"Side{i}", (sign * (w - thick) / 2, 0, h / 2), (thick, d - 2 * thick, h), color, parent=parent)
        return parent

    def cart(self, name, xyz, width=0.55, depth=0.76):
        cart = self.group(name, xyz)
        self.box("Deck", (0, 0, 0.20), (width, depth, 0.06), "work", parent=cart)
        for i, sign in enumerate((-1, 1)):
            x = sign * (width / 2 - 0.035)
            for j, end in enumerate((-1, 1)):
                self.cylinder(
                    f"Wheel{i}{j}", (x, end * (depth / 2 - 0.10), 0.087), 0.087, 0.048, "frame", (0, 90, 0), cart
                )
            self.bar(f"Upright{i}", (x, -depth / 2 + 0.03, 0.22), (x, -depth / 2 + 0.03, 0.86), 0.022, parent=cart)
        self.bar(
            "Grip",
            (-width / 2 + 0.035, -depth / 2 + 0.03, 0.86),
            (width / 2 - 0.035, -depth / 2 + 0.03, 0.86),
            0.025,
            parent=cart,
        )
        return cart

    def animate(self, path, attribute, samples):
        """Animate an existing translation or Euler rotation in frames at 24 fps."""
        prim = self.stage.GetPrimAtPath(path)
        attr = prim.GetAttribute("xformOp:" + attribute)
        assert attr, f"Missing {attribute} on {path}"
        for frame, value in samples:
            attr.Set(value, frame)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(max(self.stage.GetEndTimeCode(), samples[-1][0]))
        self.stage.SetTimeCodesPerSecond(24)
        self.stage.SetFramesPerSecond(24)
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)

    def finish(self, overview=(1.25, -2.5, 1.3), approach=(0.16, -2.8, 0.67)):
        """Frame the task geometry before adding its ground plane and studio lighting."""
        cache = UsdGeom.BBoxCache(Usd.TimeCode(0), [UsdGeom.Tokens.default_], useExtentsHint=False)
        bounds = cache.ComputeWorldBound(self.stage.GetPrimAtPath(self.parts)).ComputeAlignedRange()
        center = bounds.GetMidpoint()
        extent = bounds.GetSize()
        for name, direction in (("Overview", overview), ("Approach", approach)):
            if name == "Approach" and self.stage.GetPrimAtPath(self.parts.AppendChild("BenchTop")):
                bounds = Gf.Range3d()
                for child in self.stage.GetPrimAtPath(self.parts).GetChildren():
                    if not child.GetName().startswith("Bench"):
                        bounds.UnionWith(cache.ComputeWorldBound(child).ComputeAlignedRange())
                center = bounds.GetMidpoint()
                extent = bounds.GetSize()
            direction = Gf.Vec3d(*direction).GetNormalized()
            eye = center + direction * (extent.GetLength() * 1.7)
            view = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1))
            corners = [view.Transform(bounds.GetCorner(i)) for i in range(8)]
            width = max(max(abs(c[0]), abs(c[1]) * 1.5) for c in corners) * 2.12
            camera = UsdGeom.Camera.Define(self.stage, "/World/" + name)
            camera.AddTransformOp().Set(view.GetInverse())
            camera.CreateProjectionAttr("orthographic")
            camera.CreateHorizontalApertureAttr(width * 10)
            camera.CreateVerticalApertureAttr(width * 10 / 1.5)
            camera.CreateClippingRangeAttr((0.01, 1000))
        ground = UsdGeom.Xform.Define(self.stage, "/World/Studio").GetPath()
        self.box(
            "Floor",
            (center[0], center[1], min(0, bounds.GetMin()[2]) - 0.031),
            (100, 100, 0.06),
            "floor",
            parent=ground,
        )
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Studio/Sky")
        dome.CreateIntensityAttr(250)
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Studio/Key")
        sun.CreateIntensityAttr(1200)
        sun.CreateAngleAttr(12)
        sun.AddRotateXYZOp().Set((28, -32, -18))
