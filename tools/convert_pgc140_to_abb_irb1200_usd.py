#!/usr/bin/env python3
"""Download the vendor PGC-140 URDF and build standalone/ABB USD assets in Isaac Sim.

The originals are retained verbatim. The simulation variant adds explicitly
modelled offset fingertips (40--90 mm aperture), not a replacement gripper.
Run with /isaac-sim/python.sh in the existing Arena Docker container.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples/examples_teleop/assets/abb_irb1200_pgc140"
COMMIT = "f59f9c2f4bc8eb116448b1d798791424bf64e337"
UPSTREAM = f"https://raw.githubusercontent.com/DH-Robotics/dh_gripper_ros/{COMMIT}/dh_pgc140_urdf/"
SOURCE_FILES = (
    "package.xml",
    "urdf/dh_pgc140_urdf.urdf",
    "meshes/base_link.STL",
    "meshes/finger1_Link.STL",
    "meshes/finger2_Link.STL",
)


def download_sources(destination: Path) -> dict:
    """Keep vendor files untouched and record their exact hashes."""
    manifest = {"repository": "https://github.com/DH-Robotics/dh_gripper_ros", "commit": COMMIT, "files": {}}
    for name in SOURCE_FILES:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            request = urllib.request.Request(UPSTREAM + name, headers={"User-Agent": "Arena-asset-converter"})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
            path.write_bytes(content)
        manifest["files"][name] = {"url": UPSTREAM + name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return manifest


def origin(parent: ET.Element, xyz, rpy=(0.0, 0.0, 0.0)):
    ET.SubElement(parent, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))


def box(link: ET.Element, name: str, center, size, rgba):
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag, name=name)
        origin(element, center)
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "box", size=" ".join(map(str, size)))
        if tag == "visual":
            material = ET.SubElement(element, "material", name=name + "_material")
            ET.SubElement(material, "color", rgba=" ".join(map(str, rgba)))


def write_xml(robot, path: Path):
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def prepare_gripper(source: Path, destination: Path) -> ET.Element:
    """Preserve vendor joint geometry; add removable offset finger tooling."""
    robot = ET.parse(source / "urdf/dh_pgc140_urdf.urdf").getroot()
    robot.set("name", "pgc140_offset_fingers")
    for link in robot.findall("link"):
        link.set("name", "pgc140_" + link.get("name"))
        for material in link.findall("visual/material"):
            material.set("name", link.get("name") + "_material")
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str((source / mesh.get("filename").split("dh_pgc140_urdf/", 1)[1]).resolve()))
    # The exported finger meshes include the stock inward-facing fingertips.
    # Offset tooling REPLACES those tips: retaining them would obstruct the
    # advertised 90 mm aperture. Keep the vendor housing and prismatic joints,
    # represent the moving carriages with explicit box geometry.
    for index in (1, 2):
        link = robot.find(f"link[@name='pgc140_finger{index}_link']")
        for element in list(link):
            if element.tag in {"visual", "collision"}:
                link.remove(element)
        box(link, f"carriage{index}", (0, 0, -0.002), (0.020, 0.025, 0.016), (0.6, 0.6, 0.65, 1))
    for joint in robot.findall("joint"):
        joint.set("name", "pgc140_" + joint.get("name"))
        for tag in ("parent", "child"):
            element = joint.find(tag)
            element.set("link", "pgc140_" + element.get("link"))
        # Both physical sliders are actuated with the same position command.
        # This is an explicit two-actuator approximation of the internal rack;
        # there is no claim that software targets impose a hard mimic constraint.
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
        joint.find("limit").set("velocity", "0.0333333333")  # 25 mm / 0.75 s
        joint.find("origin").set("rpy", f"0 0 {(-1 if 'finger1' in joint.get('name') else 1) * math.pi / 2}")

    # Vendor CAD-export mass omits motor/electronics (0.34365 kg vs 1 kg
    # complete product). Scale the base inertia with its mass, retaining COM.
    base_inertial = robot.find("link[@name='pgc140_base_link']/inertial")
    scale = 0.971444 / float(base_inertial.find("mass").get("value"))
    base_inertial.find("mass").set("value", "0.971444")
    inertia = base_inertial.find("inertia")
    for key, value in list(inertia.attrib.items()):
        inertia.set(key, str(float(value) * scale))

    for index in (1, 2):
        # Slider frames rotate +/-90deg about Z. In either slider's frame,
        # x=-22.5 mm puts the pad centre at y=+/-(49-q) mm in base frame.
        # Pad half thickness=4 mm -> inner faces +/-(45-q) mm.
        name = f"pgc140_pad{index}"
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        origin(inertial, (0.0, 0.0, -0.010))
        ET.SubElement(inertial, "mass", value="0.04")
        ET.SubElement(inertial, "inertia", ixx="0.000014", ixy="0", ixz="0", iyy="0.000010", iyz="0", izz="0.000007")
        box(link, f"pad{index}_rubber", (0, 0, 0), (0.008, 0.036, 0.050), (0.12, 0.15, 0.18, 1))
        box(link, f"pad{index}_bracket", (0.01125, 0.0066, -0.028), (0.0305, 0.036, 0.006), (0.65, 0.65, 0.68, 1))
        joint = ET.SubElement(robot, "joint", name=name + "_mount", type="fixed")
        origin(joint, (-0.0225, -0.0132, 0.037))
        ET.SubElement(joint, "parent", link=f"pgc140_finger{index}_link")
        ET.SubElement(joint, "child", link=name)
    write_xml(robot, destination)
    return robot


def combine(gripper: ET.Element, destination: Path):
    source = ROOT / "isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.urdf"
    robot = ET.parse(source).getroot()
    robot.set("name", "abb_irb1200_pgc140")
    # These three empty reference frames are unnecessary for this assembly.
    for child in list(robot):
        if (child.tag == "link" and child.get("name") in {"base", "flange", "tool0"}) or (
            child.tag == "joint" and child.get("type") == "fixed"
        ):
            robot.remove(child)
    for mesh in robot.findall(".//mesh"):
        path = mesh.get("filename")
        if path.startswith("/workspaces/isaaclab_arena/"):
            mesh.set("filename", str(ROOT / path.removeprefix("/workspaces/isaaclab_arena/")))
        assert Path(mesh.get("filename")).is_file(), mesh.get("filename")
    for joint in robot.findall("joint"):
        # ROS ABB description leaves effort=0. Set a documented simulation
        # drive cap, not a claimed manufacturer joint-torque specification.
        joint.find("limit").set("effort", "300")
    robot.extend(copy.deepcopy(list(gripper)))
    joint = ET.SubElement(robot, "joint", name="pgc140_flange_mount", type="fixed")
    origin(joint, (0, 0, 0), (0, math.pi / 2, 0))
    ET.SubElement(joint, "parent", link="link_6")
    ET.SubElement(joint, "child", link="pgc140_base_link")
    write_xml(robot, destination)


def import_usd(urdf: Path, output: Path, fixed_base: bool):
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Usd, UsdPhysics

    config = URDFImporterConfig()
    config.urdf_path = str(urdf)
    config.usd_path = str(output)
    config.make_default_prim = True
    config.fix_base = fixed_base
    config.merge_mesh = False
    config.collision_from_visuals = False
    config.self_collision = False
    config.import_inertia_tensor = True
    generated = Path(URDFImporter(config).import_urdf())
    stage = Usd.Stage.Open(str(generated))
    assert stage and stage.GetDefaultPrim().IsValid(), generated
    # A flattened stage is portable within this asset folder and convenient
    # for standalone Newton; no runtime dependency on a ROS package resolver.
    target = output / (urdf.stem + ".usda")
    stage.Flatten().Export(str(target))
    joint_names = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    assert len(joint_names) == 2, joint_names
    print(f"[USD] {target}; prismatic joints: {joint_names}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    parser.add_argument("--prepare-only", action="store_true", help="Download/normalize URDF without starting Kit.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "vendor"
    manifest = download_sources(source)
    standalone = output / "pgc140_offset_fingers.urdf"
    combined = output / "abb_irb1200_pgc140.urdf"
    gripper = prepare_gripper(source, standalone)
    combine(gripper, combined)
    manifest.update({
        "specification_url": "https://www.dh-robotics.com/product/pgc",
        "license_note": "Vendor package.xml declares BSD; retain upstream files; no standalone license text supplied.",
        "modifications": ["prefix link/joint names", "two position-driven sliders, no mimic",
                          "stock fingers replaced by box carriages and 40-90 mm offset tooling, 0.08 kg added",
                          "base mass scaled to 1 kg complete gripper excluding tooling",
                          "ABB link_6 mount Ry(pi/2); pad TCP = (0.130,0,0) in link_6"],
    })
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[URDF] {standalone}\n[URDF] {combined}", flush=True)
    if args.prepare_only:
        return

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        import_usd(standalone, output / "gripper", False)
        import_usd(combined, output / "combined", True)
    finally:
        app.close()


if __name__ == "__main__":
    main()