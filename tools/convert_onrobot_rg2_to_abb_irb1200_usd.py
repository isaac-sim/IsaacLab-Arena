#!/usr/bin/env python3
"""Prepare an ABB IRB1200 + OnRobot RG2 USD for Newton standalone examples.

Run inside the IsaacLab-Arena Docker workspace:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh tools/convert_onrobot_rg2_to_abb_irb1200_usd.py --headless

The source RG2 description is expected at:

    examples/examples_teleop/assets/onrobot_rg2_description

It was copied from:

    https://github.com/ABC-iRobotics/onrobot-ros2/tree/main/onrobot_rg_description
"""

from __future__ import annotations

import argparse
import os
import shutil
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


def _add_boxlike_inertial(link: ET.Element, mass: str = "0.05") -> None:
    inertial = ET.SubElement(link, "inertial")
    inertial.append(_origin("0 0 0"))
    ET.SubElement(inertial, "mass", {"value": mass})
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


def _add_mesh_link(robot: ET.Element, name: str, visual_mesh: Path, collision_mesh: Path, color: str) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    _add_boxlike_inertial(link, "0.7" if name == "rg2_base_link" else "0.05")

    visual = ET.SubElement(link, "visual")
    visual.append(_origin("0 0 0"))
    visual.append(_mesh(visual_mesh))
    material = ET.SubElement(visual, "material", {"name": f"{name}_mat"})
    ET.SubElement(material, "color", {"rgba": color})

    collision = ET.SubElement(link, "collision")
    collision.append(_origin("0 0 0"))
    collision.append(_mesh(collision_mesh))


def _add_revolute_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    xyz: str,
    rpy: str,
    axis: str,
    lower: str,
    upper: str,
    mimic: tuple[str, str] | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
    joint.append(_origin(xyz, rpy))
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(joint, "limit", {"lower": lower, "upper": upper, "velocity": "100.0", "effort": "1000"})
    if mimic is not None:
        ET.SubElement(joint, "mimic", {"joint": mimic[0], "multiplier": mimic[1], "offset": "0"})


def write_rg2_urdf(source_dir: Path, output_urdf: Path) -> Path:
    visual = source_dir / "meshes/rg2/visual"
    collision = source_dir / "meshes/rg2/collision"
    for required in [
        visual / "base_link.stl",
        visual / "outer_knuckle.stl",
        visual / "inner_knuckle.stl",
        visual / "inner_finger.stl",
        collision / "base_link.stl",
        collision / "outer_knuckle.stl",
        collision / "inner_knuckle.stl",
        collision / "inner_finger.stl",
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    robot = ET.Element("robot", {"name": "onrobot_rg2"})
    _add_mesh_link(robot, "rg2_base_link", visual / "base_link.stl", collision / "base_link.stl", "0.8 0.8 0.8 1")

    for side in ("left", "right"):
        _add_mesh_link(
            robot,
            f"{side}_outer_knuckle",
            visual / "outer_knuckle.stl",
            collision / "outer_knuckle.stl",
            "0.8 0.8 0.8 1",
        )
        _add_mesh_link(
            robot,
            f"{side}_inner_knuckle",
            visual / "inner_knuckle.stl",
            collision / "inner_knuckle.stl",
            "0.8 0.8 0.8 1",
        )
        _add_mesh_link(
            robot,
            f"{side}_inner_finger",
            visual / "inner_finger.stl",
            collision / "inner_finger.stl",
            "0.1 0.1 0.1 1",
        )

    _add_revolute_joint(
        robot,
        "finger_joint",
        "rg2_base_link",
        "left_outer_knuckle",
        "0 -0.017178 0.125797",
        "0 0 0",
        "-1 0 0",
        "-0.558505",
        "0.785398",
    )
    _add_revolute_joint(
        robot,
        "left_inner_knuckle_joint",
        "rg2_base_link",
        "left_inner_knuckle",
        "0 -0.007678 0.142297",
        "0 0 0",
        "1 0 0",
        "-0.785398",
        "0.785398",
        mimic=("finger_joint", "-1"),
    )
    _add_revolute_joint(
        robot,
        "left_inner_finger_joint",
        "left_outer_knuckle",
        "left_inner_finger",
        "0 -0.039592 0.038177",
        "0 0 0",
        "1 0 0",
        "-0.872665",
        "0.872665",
        mimic=("finger_joint", "1"),
    )
    _add_revolute_joint(
        robot,
        "right_outer_knuckle_joint",
        "rg2_base_link",
        "right_outer_knuckle",
        "0 0.017178 0.125797",
        "0 0 3.141592653589793",
        "1 0 0",
        "-0.785398",
        "0.785398",
        mimic=("finger_joint", "-1"),
    )
    _add_revolute_joint(
        robot,
        "right_inner_knuckle_joint",
        "rg2_base_link",
        "right_inner_knuckle",
        "0 0.007678 0.142297",
        "0 0 -3.141592653589793",
        "1 0 0",
        "-0.785398",
        "0.785398",
        mimic=("finger_joint", "-1"),
    )
    _add_revolute_joint(
        robot,
        "right_inner_finger_joint",
        "right_outer_knuckle",
        "right_inner_finger",
        "0 -0.039592 0.038177",
        "0 0 0",
        "1 0 0",
        "-0.872665",
        "0.872665",
        mimic=("finger_joint", "1"),
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
    ET.SubElement(fixed_joint, "child", {"link": "rg2_base_link"})

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(abb_robot)
    ET.indent(tree, space="  ")
    tree.write(str(output_urdf), encoding="utf-8", xml_declaration=True)
    return output_urdf


def patch_rg2_usd_for_newton(usd_path: Path) -> None:
    text = usd_path.read_text(encoding="utf-8")
    text = text.replace(
        'def Xform "onrobot_rg2" (\n'
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]\n'
        '    prepend references = @./payloads/base.usda@',
        'def Xform "onrobot_rg2" (\n'
        '    prepend references = @./payloads/base.usda@',
    )
    usd_path.write_text(text, encoding="utf-8")

    physics_path = usd_path.parent / "payloads/Physics/physics.usda"
    if physics_path.exists():
        physics_text = physics_path.read_text(encoding="utf-8")
        physics_text = physics_text.replace(
            '            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI", '
            '"NewtonArticulationRootAPI", "PhysicsMassAPI"]',
            '            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        )
        physics_path.write_text(physics_text, encoding="utf-8")


def write_abb_rg2_combined_usd(
    abb_source_dir: Path,
    combined_dir: Path,
    rg2_usd_path: Path,
    mount_translate: tuple[float, float, float],
    mount_orient: tuple[float, float, float, float],
) -> Path:
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    shutil.copytree(abb_source_dir, combined_dir)

    combined_usd_path = combined_dir / "irb1200_7_70.usda"
    default_prim_name = "abb_irb1200_7_70"
    rel_rg2 = Path(os.path.relpath(rg2_usd_path, combined_usd_path.parent))
    text = combined_usd_path.read_text(encoding="utf-8")
    insert = f'''
                over "Geometry"
                {{
                    over "base_link"
                    {{
                        over "link_1"
                        {{
                            over "link_2"
                            {{
                                over "link_3"
                                {{
                                    over "link_4"
                                    {{
                                        over "link_5"
                                        {{
                                            over "link_6"
                                            {{
                                                def Xform "onrobot_rg2_mount" (
                                                    delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI"]
                                                    prepend references = @{rel_rg2.as_posix()}@
                                                )
                                                {{
                                                    quatd xformOp:orient = {mount_orient}
                                                    double3 xformOp:translate = {mount_translate}
                                                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                                                }}

                                                def PhysicsFixedJoint "onrobot_rg2_fixed_joint"
                                                {{
                                                    rel physics:body0 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6>
                                                    rel physics:body1 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/onrobot_rg2_mount/onrobot_rg2/Geometry/rg2_base_link>
                                                    point3f physics:localPos0 = {mount_translate}
                                                    point3f physics:localPos1 = (0, 0, 0)
                                                    quatf physics:localRot0 = {mount_orient}
                                                    quatf physics:localRot1 = (1, 0, 0, 0)
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
'''
    marker = f'def Xform "{default_prim_name}" (\n'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find default prim block in {combined_usd_path}")
    body_start = text.find(")\n{", start)
    if body_start < 0:
        raise RuntimeError(f"Could not find default prim opening brace in {combined_usd_path}")
    open_brace = body_start + 2
    text = text[: open_brace + 1] + insert + text[open_brace + 1 :]
    combined_usd_path.write_text(text, encoding="utf-8")
    return combined_usd_path


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
        default=Path("examples/examples_teleop/assets/onrobot_rg2_usd"),
        help="Output directory for the standalone RG2 USD.",
    )
    parser.add_argument(
        "--abb-source-dir",
        type=Path,
        default=Path("isaaclab_arena/assets/robots/abb/irb1200_7_70"),
        help="Existing converted ABB IRB1200 USD directory to copy.",
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
        default=Path("examples/examples_teleop/assets/abb_irb1200_onrobot_rg2_imported_newton"),
        help="Output directory for the assembled ABB + RG2 USD.",
    )
    parser.add_argument("--mount-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--mount-orient", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--mount-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    source_dir = (repo_root / args.source_dir).resolve()
    rg2_usd_dir = (repo_root / args.rg2_usd_dir).resolve()
    abb_source_dir = (repo_root / args.abb_source_dir).resolve()
    abb_urdf_path = (repo_root / args.abb_urdf).resolve()
    combined_dir = (repo_root / args.combined_dir).resolve()
    urdf_path = source_dir / "onrobot_rg2_generated.urdf"
    combined_urdf_path = combined_dir / "irb1200_7_70_onrobot_rg2.urdf"

    print(f"[INFO] writing RG2 URDF: {urdf_path}")
    write_rg2_urdf(source_dir, urdf_path)

    sim_app = SimulationApp({"headless": args.headless})
    try:
        print(f"[INFO] importing RG2 URDF to USD dir: {rg2_usd_dir}")
        rg2_usd_path = import_urdf_to_usd(urdf_path, rg2_usd_dir)
        patch_rg2_usd_for_newton(rg2_usd_path)
        print(f"[INFO] imported RG2 USD: {rg2_usd_path}")

        print(f"[INFO] writing combined ABB + RG2 URDF: {combined_urdf_path}")
        write_abb_rg2_combined_urdf(
            abb_urdf_path=abb_urdf_path,
            rg2_urdf_path=urdf_path,
            output_urdf=combined_urdf_path,
            mount_translate=tuple(args.mount_translate),
            mount_rpy=tuple(args.mount_rpy),
        )

        print(f"[INFO] importing combined ABB + RG2 URDF to USD dir: {combined_dir}")
        combined_usd = import_urdf_to_usd(combined_urdf_path, combined_dir)
        print(f"[INFO] wrote ABB + RG2 USD: {combined_usd}")
        print(combined_usd)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
