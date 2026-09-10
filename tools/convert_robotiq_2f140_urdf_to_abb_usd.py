#!/usr/bin/env python3
"""Download ROS-Industrial 2F-140 xacro, convert gripper USD, compose with local ABB USD.

Run with the existing Docker /isaac-sim/python.sh. Requires xacro==2.1.1.
Vendor originals, all six revolute joints, all five mimic relations, and stock
pad geometry are retained. No prismatic replacement or PGC-style tooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples/examples_teleop/assets/abb_irb1200_robotiq_2f140_urdf"
COMMIT = "45196f6558fe8ba9d89bc8a105396c68c3e7e892"
PACKAGE = "robotiq_2f_140_gripper_visualization"
BASE_URL = f"https://raw.githubusercontent.com/ros-industrial-attic/robotiq/{COMMIT}/"


def download_sources(destination):
    files = ["LICENSE", f"{PACKAGE}/package.xml", f"{PACKAGE}/README.md"]
    files += [f"{PACKAGE}/urdf/{name}" for name in (
        "robotiq_arg2f.xacro", "robotiq_arg2f_140_model.xacro",
        "robotiq_arg2f_140_model_macro.xacro", "robotiq_arg2f_transmission.xacro")]
    files += [f"{PACKAGE}/meshes/{kind}/{name}.stl" for kind in ("visual", "collision")
              for name in ("robotiq_arg2f_base_link", "robotiq_arg2f_140_outer_knuckle",
                           "robotiq_arg2f_140_outer_finger", "robotiq_arg2f_140_inner_knuckle",
                           "robotiq_arg2f_140_inner_finger")]

    def fetch(name):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Arena-URDF-converter"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=25) as response:
                        content = response.read()
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    print(f"[RETRY] {name} ({attempt + 1}/3)", flush=True)
            path.write_bytes(content)
        print(f"[SOURCE] {name}", flush=True)
        return name, {"url": BASE_URL + name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as executor:
        return dict(executor.map(fetch, files))


def expand_urdf(output):
    import xacro

    source = output / "vendor" / PACKAGE
    resolved = output / "resolved_xacro"
    resolved.mkdir(exist_ok=True)
    for file in (source / "urdf").glob("*.xacro"):
        text = file.read_text().replace(f"$(find {PACKAGE})/urdf/", str(resolved) + "/")
        (resolved / file.name).write_text(text)
    document = xacro.process_file(str(resolved / "robotiq_arg2f_140_model.xacro"))
    robot = ET.fromstring(document.toxml())
    robot.set("name", "robotiq_2f140")
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str(source / mesh.get("filename").removeprefix(f"package://{PACKAGE}/")))
        assert Path(mesh.get("filename")).is_file(), mesh.attrib
    for link in robot.findall("link"):
        for material in link.findall("visual/material"):
            material.set("name", link.get("name") + "_material")
        # Pads have no inertia in upstream. Give each stock rubber pad 10 g
        # with box inertia, rather than letting an importer assign arbitrary mass.
        if link.get("name").endswith("_pad"):
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.01")
            a, b, c = 0.03, 0.07, 0.0075
            ET.SubElement(inertial, "inertia", ixx=str(0.01 * (b*b+c*c)/12),
                          iyy=str(0.01 * (a*a+c*c)/12), izz=str(0.01 * (a*a+b*b)/12),
                          ixy="0", ixz="0", iyz="0")
    for transmission in robot.findall("transmission"):
        robot.remove(transmission)
    mimic = {}
    for joint in robot.findall("joint"):
        relation = joint.find("mimic")
        if relation is not None:
            mimic[joint.get("name")] = dict(leader=relation.get("joint"),
                                           multiplier=float(relation.get("multiplier", "1")),
                                           offset=float(relation.get("offset", "0")))
    assert len(mimic) == 5, mimic
    # Keep upstream masses, inertias and effort metadata intact (except pads).
    # The example sets realistic bounded actuator settings, not 1000 Nm defaults.
    path = output / "robotiq_2f140.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
    return path, mimic


def convert_gripper(urdf, output):
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Usd

    config = URDFImporterConfig()
    config.urdf_path = str(urdf)
    config.usd_path = str(output)
    config.fix_base = False
    config.make_default_prim = True
    config.merge_mesh = False
    config.collision_from_visuals = False
    config.self_collision = False
    config.import_inertia_tensor = True
    for option in ("parse_mimic", "import_mimic"):
        if hasattr(config, option):
            setattr(config, option, True)
    imported = Path(URDFImporter(config).import_urdf())
    stage = Usd.Stage.Open(str(imported))
    target = output / "robotiq_2f140.usda"
    stage.Flatten().Export(str(target))
    return target


def compose_usd(abb_path, gripper_path, destination):
    """Reference BOTH existing USD assets, weld at link_6, then flatten locally."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    abb = Usd.Stage.Open(str(abb_path))
    source_root = abb.GetDefaultPrim().GetPath()
    assembly_path = destination / "assembly.usda"
    stage = Usd.Stage.CreateNew(str(assembly_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/abb_irb1200_2f140", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(abb_path, destination), source_root)
    flange = next(p for p in stage.Traverse() if p.GetName() == "link_6" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    mount = stage.DefinePrim(flange.GetPath().AppendChild("robotiq_mount"), "Xform")
    mount.GetReferences().AddReference(os.path.relpath(gripper_path, destination))
    mount_rotation = Gf.Quatd(math.sqrt(0.5), 0, math.sqrt(0.5), 0)
    xf = UsdGeom.Xformable(mount)
    xf.ClearXformOpOrder()
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(mount_rotation)
    descendants = list(Usd.PrimRange(mount))
    for prim in descendants:
        if prim.IsInstance():
            prim.SetInstanceable(False)
        for schema in prim.GetAppliedSchemas():
            if "Articulation" in schema:
                prim.RemoveAppliedSchema(schema)
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                joint.CreateJointEnabledAttr(False)
    base = next(p for p in Usd.PrimRange(mount) if p.GetName() == "robotiq_arg2f_base_link")
    fixed = UsdPhysics.FixedJoint.Define(stage, root.GetPath().AppendPath("Physics/robotiq_mount_joint"))
    fixed.CreateBody0Rel().SetTargets([flange.GetPath()])
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(mount_rotation))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1))
    stage.GetRootLayer().Save()
    target = destination / "abb_irb1200_robotiq_2f140.usda"
    stage.Flatten().Export(str(target))
    final = Usd.Stage.Open(str(target))
    assert len([p for p in final.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 12
    assert not [p for p in final.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    print(f"[USD] {gripper_path}\n[USD] {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    parser.add_argument("--abb-usd", type=Path, default=ROOT / "isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.usda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository="https://github.com/ros-industrial-attic/robotiq", commit=COMMIT,
                    files=download_sources(output / "vendor"))
    urdf, mimic = expand_urdf(output)
    manifest.update(mimic=mimic, abb_usd=str(args.abb_usd.resolve()),
                    modifications=["absolute mesh paths and nonempty material names", "omit ROS transmission",
                                   "10g box inertial per stock pad", "USD fixed mount at ABB link_6, Ry(pi/2)"])
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[URDF] {urdf}; mimic={mimic}", flush=True)
    if args.prepare_only:
        return
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    exit_code = 0
    try:
        gripper = convert_gripper(urdf, output / "gripper")
        compose_usd(args.abb_usd.resolve(), gripper, output)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        # Kit 6.0.1 can abort while destroying an importer TaskGroup during
        # ordinary cleanup. USD exports above are synchronous and saved;
        # use the supported immediate-exit API, preserving conversion failures.
        app.close(skip_cleanup=True, exit_code=exit_code)


if __name__ == "__main__":
    main()