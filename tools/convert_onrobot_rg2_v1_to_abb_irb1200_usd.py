#!/usr/bin/env python3
"""Assemble an ABB IRB1200 + OnRobot RG2 (v1) USD for Newton standalone examples.

Run inside the IsaacLab-Arena Docker workspace:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh tools/convert_onrobot_rg2_v1_to_abb_irb1200_usd.py --headless

Unlike the older ``convert_onrobot_rg2_to_abb_irb1200_usd.py`` (which reproduced
the legacy inner/outer-knuckle RG2), this script reproduces the current official
description from inria-paris-robotics-lab/onrobot_ros:

    onrobot_description/urdf/onrobot_rg.urdf.xacro  +  config/rg2_v1.yaml

The RG2 v1 is a revolute-driven parallelogram gripper: a master joint
``gripper_joint`` (revolute about Y, 0..1.3 rad) plus five mimic joints keep the
two finger tips parallel while opening/closing. The finger tips / flex fingers
are the real grasp contact surfaces.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp


def _set_if_present(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _mesh(path: Path) -> ET.Element:
    geometry = ET.Element("geometry")
    ET.SubElement(geometry, "mesh", {"filename": str(path.resolve())})
    return geometry


def _origin(xyz: str, rpy: str = "0 0 0") -> ET.Element:
    return ET.Element("origin", {"xyz": xyz, "rpy": rpy})


def _inertial(link: ET.Element, mass: float) -> None:
    inertial = ET.SubElement(link, "inertial")
    inertial.append(_origin("0 0 0"))
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1.0e-03",
            "ixy": "1.0e-06",
            "ixz": "1.0e-06",
            "iyy": "1.0e-03",
            "iyz": "1.0e-06",
            "izz": "1.0e-03",
        },
    )


def _mesh_link(robot: ET.Element, name: str, mesh_file: Path, mass: float, color: str) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    _inertial(link, mass)

    visual = ET.SubElement(link, "visual")
    visual.append(_origin("0 0 0"))
    visual.append(_mesh(mesh_file))
    material = ET.SubElement(visual, "material", {"name": f"{name}_mat"})
    ET.SubElement(material, "color", {"rgba": color})

    collision = ET.SubElement(link, "collision")
    collision.append(_origin("0 0 0"))
    collision.append(_mesh(mesh_file))


def _frame_link(robot: ET.Element, name: str, mass: float = 0.001) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    _inertial(link, mass)


def _fixed_joint(robot: ET.Element, name: str, parent: str, child: str, xyz: str, rpy: str = "0 0 0") -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    joint.append(_origin(xyz, rpy))
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def _revolute_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: str,
    rpy: str,
    axis: str,
    limit: dict[str, float],
    mimic: tuple[str, str] | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
    joint.append(_origin(xyz, rpy))
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": str(limit["lower"]),
            "upper": str(limit["upper"]),
            "velocity": str(limit["velocity"]),
            "effort": str(limit["effort"]),
        },
    )
    if mimic is not None:
        ET.SubElement(joint, "mimic", {"joint": mimic[0], "multiplier": mimic[1], "offset": "0"})


def write_rg2_urdf(source_dir: Path, output_urdf: Path) -> Path:
    """Generate the OnRobot RG2 v1 URDF (moment_arm/truss_arm parallel gripper)."""
    visual = source_dir / "meshes/rg2_v1/visual"
    for required in ["body", "single_bracket", "moment_arm", "truss_arm", "finger_tip", "flex_finger"]:
        if not (visual / f"{required}.stl").exists():
            raise FileNotFoundError(visual / f"{required}.stl")

    # config/rg2_v1.yaml values, kept inline to avoid a yaml dependency here.
    limit = {"lower": 0.0, "upper": 1.3, "effort": 10.0, "velocity": 50.0}
    origin = {
        "single_bracket": "0.0 0.0 0.05",
        "moment_arm": "-0.017 0.0 0.055",
        "truss_arm": "-0.0075 0.0 0.0715",
        "finger_tip": "-0.0256 0.0 0.04868",
        "flex_finger": "0.03059 0.0 0.0172",
        "grasp_frame": "0.0 0.0 0.14",
    }
    mass = {
        "body": 0.65,
        "single_bracket": 0.09,
        "moment_arm": 0.001,
        "truss_arm": 0.001,
        "finger_tip": 0.001,
        "flex_finger": 0.001,
    }
    offset = -0.772  # kinematics.offset, fixes the parallelogram linkage angle

    robot = ET.Element("robot", {"name": "onrobot_rg2"})

    # Body chain: bracket (root, mounted to ABB link_6) -> body -> grasp_frame.
    _mesh_link(robot, "gripper_bracket", visual / "single_bracket.stl", mass["single_bracket"], "0.8 0.8 0.8 1")
    _mesh_link(robot, "gripper_body", visual / "body.stl", mass["body"], "0.8 0.8 0.8 1")
    _frame_link(robot, "gripper_grasp_frame")

    for finger in ("finger_1", "finger_2"):
        _frame_link(robot, f"gripper_{finger}_origin")
        _mesh_link(robot, f"gripper_{finger}_moment_arm", visual / "moment_arm.stl", mass["moment_arm"], "0.8 0.8 0.8 1")
        _mesh_link(robot, f"gripper_{finger}_truss_arm", visual / "truss_arm.stl", mass["truss_arm"], "0.8 0.8 0.8 1")
        _mesh_link(robot, f"gripper_{finger}_finger_tip", visual / "finger_tip.stl", mass["finger_tip"], "0.1 0.1 0.1 1")
        _mesh_link(robot, f"gripper_{finger}_flex_finger", visual / "flex_finger.stl", mass["flex_finger"], "0.1 0.1 0.1 1")

    _fixed_joint(robot, "gripper_body_joint", "gripper_bracket", "gripper_body", origin["single_bracket"])
    _fixed_joint(robot, "gripper_grasp_frame_joint", "gripper_body", "gripper_grasp_frame", origin["grasp_frame"])

    for finger, is_master in (("finger_1", True), ("finger_2", False)):
        _fixed_joint(
            robot,
            f"gripper_{finger}_origin_joint",
            "gripper_body",
            f"gripper_{finger}_origin",
            "0 0 0",
            "0 0 0" if is_master else "0 0 3.141592653589793",
        )
        joint_name = "gripper_joint" if is_master else "gripper_mirror_joint"
        # NOTE: no <mimic> tags. The five follower joints are driven in software
        # (same target as gripper_joint) because Newton's URDF importer mangles
        # the joint tree when mimic is present, and Newton solves mimic
        # unreliably. multiplier is 1 / offset 0 for every follower, so the
        # follower target simply equals the master target.
        _revolute_joint(
            robot,
            joint_name,
            f"gripper_{finger}_origin",
            f"gripper_{finger}_moment_arm",
            origin["moment_arm"],
            f"0 {offset} 0",
            "0 1 0",
            limit,
        )
        _revolute_joint(
            robot,
            f"gripper_{finger}_truss_arm_joint",
            f"gripper_{finger}_origin",
            f"gripper_{finger}_truss_arm",
            origin["truss_arm"],
            f"0 {offset} 0",
            "0 1 0",
            limit,
        )
        _revolute_joint(
            robot,
            f"gripper_{finger}_finger_tip_joint",
            f"gripper_{finger}_truss_arm",
            f"gripper_{finger}_finger_tip",
            origin["finger_tip"],
            f"0 {-offset} 0",
            "0 -1 0",
            limit,
        )
        _fixed_joint(
            robot,
            f"gripper_{finger}_flex_finger_joint",
            f"gripper_{finger}_finger_tip",
            f"gripper_{finger}_flex_finger",
            origin["flex_finger"],
        )

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(str(output_urdf), encoding="utf-8", xml_declaration=True)
    return output_urdf


def import_urdf_to_usd(urdf_path: Path, usd_output_dir: Path) -> Path:
    import omni.kit.app
    import omni.usd
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    usd_output_dir.mkdir(parents=True, exist_ok=True)
    config = URDFImporterConfig()
    config.urdf_path = str(urdf_path)
    config.usd_path = str(usd_output_dir)
    config.make_default_prim = True
    config.fix_base = False
    config.merge_mesh = False
    config.collision_from_visuals = False
    config.self_collision = False
    config.import_inertia_tensor = True
    _set_if_present(config, "parse_mimic", True)
    _set_if_present(config, "import_mimic", True)
    _set_if_present(config, "default_drive_type", "position")

    importer = URDFImporter(config)
    usd_path = Path(importer.import_urdf()).resolve()
    omni.usd.get_context().open_stage(str(usd_path))
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Failed to import RG2 USD from {urdf_path}")
    stage.GetRootLayer().Save()
    return usd_path


def write_abb_rg2_combined_urdf(
    abb_urdf_path: Path,
    rg2_urdf_path: Path,
    output_urdf: Path,
    mount_translate: tuple[float, float, float],
    mount_rpy: tuple[float, float, float],
) -> Path:
    abb_tree = ET.parse(abb_urdf_path)
    abb_robot = abb_tree.getroot()
    rg2_robot = ET.parse(rg2_urdf_path).getroot()

    abb_robot.set("name", "abb_irb1200_7_70_onrobot_rg2")
    for child in list(rg2_robot):
        if child.tag in {"link", "joint", "material"}:
            abb_robot.append(child)

    fixed_joint = ET.SubElement(abb_robot, "joint", {"name": "onrobot_rg2_fixed_joint", "type": "fixed"})
    fixed_joint.append(
        _origin(
            " ".join(str(v) for v in mount_translate),
            " ".join(str(v) for v in mount_rpy),
        )
    )
    ET.SubElement(fixed_joint, "parent", {"link": "link_6"})
    ET.SubElement(fixed_joint, "child", {"link": "gripper_bracket"})

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(abb_robot)
    ET.indent(tree, space="  ")
    tree.write(str(output_urdf), encoding="utf-8", xml_declaration=True)
    return output_urdf


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("examples/examples_teleop/assets/onrobot_rg2_description"),
        help="Directory containing the copied onrobot_rg_description package contents.",
    )
    parser.add_argument(
        "--rg2-usd-dir",
        type=Path,
        default=Path("examples/examples_teleop/assets/onrobot_rg2_usd_v1"),
        help="Output directory for the standalone RG2 v1 USD.",
    )
    parser.add_argument(
        "--abb-urdf",
        type=Path,
        default=Path("isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.urdf"),
        help="Existing expanded ABB IRB1200 URDF.",
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=Path("examples/examples_teleop/assets/abb_irb1200_onrobot_rg2_v1_newton"),
        help="Output directory for the assembled ABB + RG2 v1 USD.",
    )
    parser.add_argument("--mount-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    # rpy=(pi, 0, 0): flip the fingers from +Z (up) to -Z (down) directly at the
    # mount so the fingers point down in the link_6 home pose. The grasp contact
    # faces (finger_tip local +Y) already lie in the world Y plane; no Z-rotation.
    parser.add_argument(
        "--mount-rpy",
        type=float,
        nargs=3,
        default=(3.141592653589793, 0.0, 0.0),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    source_dir = (repo_root / args.source_dir).resolve()
    rg2_usd_dir = (repo_root / args.rg2_usd_dir).resolve()
    abb_urdf_path = (repo_root / args.abb_urdf).resolve()
    combined_dir = (repo_root / args.combined_dir).resolve()
    urdf_path = source_dir / "onrobot_rg2_v1_generated.urdf"
    combined_urdf_path = combined_dir / "irb1200_7_70_onrobot_rg2.urdf"

    print(f"[INFO] writing RG2 v1 URDF: {urdf_path}")
    write_rg2_urdf(source_dir, urdf_path)

    sim_app = SimulationApp({"headless": args.headless})
    try:
        print(f"[INFO] importing RG2 v1 URDF to USD dir: {rg2_usd_dir}")
        rg2_usd_path = import_urdf_to_usd(urdf_path, rg2_usd_dir)
        print(f"[INFO] imported RG2 v1 USD: {rg2_usd_path}")

        print(f"[INFO] writing combined ABB + RG2 v1 URDF: {combined_urdf_path}")
        write_abb_rg2_combined_urdf(
            abb_urdf_path=abb_urdf_path,
            rg2_urdf_path=urdf_path,
            output_urdf=combined_urdf_path,
            mount_translate=tuple(args.mount_translate),
            mount_rpy=tuple(args.mount_rpy),
        )

        print(f"[INFO] importing combined ABB + RG2 v1 URDF to USD dir: {combined_dir}")
        combined_usd = import_urdf_to_usd(combined_urdf_path, combined_dir)
        print(f"[INFO] wrote ABB + RG2 v1 USD: {combined_usd}")
        print(combined_usd)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
