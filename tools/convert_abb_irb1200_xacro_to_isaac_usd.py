#!/usr/bin/env python3
"""Convert ABB IRB1200-7/0.70 xacro into an Isaac Lab friendly USD.

Run inside the IsaacLab-Arena Docker workspace:

    /isaac-sim/python.sh tools/convert_abb_irb1200_xacro_to_isaac_usd.py
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

from isaacsim import SimulationApp


def _load_xacro_expander(repo_root: Path):
    module_path = repo_root / "tools/abb_xacro_to_urdf.py"
    spec = importlib.util.spec_from_file_location("abb_xacro_to_urdf", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load xacro expander from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rewrite_package_mesh_paths(urdf_path: Path, abb_root: Path) -> None:
    text = urdf_path.read_text(encoding="utf-8")
    package_root = (abb_root / "abb_irb1200_support").resolve()
    text = text.replace("package://abb_irb1200_support/", f"{package_root}/")
    urdf_path.write_text(text, encoding="utf-8")


def _expand_xacro_to_urdf(repo_root: Path, xacro_path: Path, abb_root: Path, output_urdf: Path) -> Path:
    expander = _load_xacro_expander(repo_root)
    colors = expander._load_color_properties(abb_root / "abb_resources/urdf/common_colours.xacro")
    robot_name, macro_name, macro_path = expander._entry_info(xacro_path)
    robot = expander._expand_macro(macro_path, robot_name, macro_name, colors)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = expander.ET.ElementTree(robot)
    expander.ET.indent(tree, space="  ")
    tree.write(str(output_urdf), encoding="utf-8", xml_declaration=True)
    _rewrite_package_mesh_paths(output_urdf, abb_root)
    return output_urdf


def _set_if_present(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _patch_articulation_root_for_isaac_lab(usd_path: Path, default_prim_name: str) -> None:
    root_text = usd_path.read_text(encoding="utf-8")
    root_text = root_text.replace(
        f'def Xform "{default_prim_name}" (\n'
        "    prepend references = @./payloads/base.usda@",
        f'def Xform "{default_prim_name}" (\n'
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]\n'
        "    prepend references = @./payloads/base.usda@",
    )
    usd_path.write_text(root_text, encoding="utf-8")

    physics_path = usd_path.parent / "payloads/Physics/physics.usda"
    physics_text = physics_path.read_text(encoding="utf-8")
    physics_text = physics_text.replace(
        '    over "Geometry" (\n'
        '        prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]\n'
        "    )",
        '    over "Geometry"',
    )
    physics_path.write_text(physics_text, encoding="utf-8")
    print("[INFO] moved articulation root API from Geometry to default prim")


def _create_robotiq_control_copy(usd_path: Path, default_prim_name: str, suffix: str) -> Path:
    control_dir = usd_path.parent.parent / f"{usd_path.parent.name}_robotiq_2f140_{suffix}"
    if control_dir.exists():
        shutil.rmtree(control_dir)
    shutil.copytree(usd_path.parent, control_dir)

    control_usd_path = control_dir / usd_path.name
    text = control_usd_path.read_text(encoding="utf-8")
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
                                                def Xform "robotiq_2f140_mount" (
                                                    delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
                                                    prepend references = @../../../grippers/robotiq_2f140/Robotiq_2F_140_physics_edit.usd@
                                                )
                                                {{
                                                    quatd xformOp:orient = (0, 0.7071067811865476, 0, 0.7071067811865475)
                                                    double3 xformOp:scale = (1, 1, 1)
                                                    double3 xformOp:translate = (-0.005, 0, 0)
                                                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

                                                }}

                                                def PhysicsFixedJoint "robotiq_2f140_fixed_joint"
                                                {{
                                                    rel physics:body0 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6>
                                                    rel physics:body1 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/robotiq_2f140_mount/robotiq_base_link>
                                                    point3f physics:localPos0 = (-0.005, 0, 0)
                                                    point3f physics:localPos1 = (0, 0, 0)
                                                    quatf physics:localRot0 = (0, 0.70710677, 0, 0.70710677)
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
        raise RuntimeError(f"Could not find default prim block in {control_usd_path}")
    body_start = text.find(")\n{", start)
    if body_start < 0:
        raise RuntimeError(f"Could not find default prim opening brace in {control_usd_path}")
    open_brace = body_start + 2
    text = text[: open_brace + 1] + insert + text[open_brace + 1 :]
    control_usd_path.write_text(text, encoding="utf-8")
    print(f"[INFO] wrote controllable Robotiq 2F-140 {suffix} copy: {control_usd_path}")
    return control_usd_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ABB IRB1200-7/0.70 xacro to Isaac Lab friendly USD.")
    parser.add_argument(
        "--xacro",
        type=Path,
        default=Path(
            "../abb/abb_irb1200_support/urdf/irb1200_7_70.xacro"
        ),
        help="Input xacro path.",
    )
    parser.add_argument(
        "--abb-root",
        type=Path,
        default=Path("../abb"),
        help="Path containing abb_resources and abb_irb1200_support.",
    )
    parser.add_argument(
        "--urdf-output",
        type=Path,
        default=Path("isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.urdf"),
        help="Expanded URDF output path.",
    )
    parser.add_argument(
        "--usd-output-dir",
        type=Path,
        default=Path("isaaclab_arena/assets/robots/abb"),
        help="Directory where Isaac Sim's URDF importer writes USD output.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    xacro_path = args.xacro.resolve()
    abb_root = args.abb_root.resolve()
    urdf_output = args.urdf_output.resolve()
    usd_output_dir = args.usd_output_dir.resolve()

    print(f"[INFO] expanding xacro: {xacro_path}")
    _expand_xacro_to_urdf(repo_root, xacro_path, abb_root, urdf_output)
    print(f"[INFO] wrote URDF: {urdf_output}")

    sim_app = SimulationApp({"headless": args.headless})
    try:
        import omni.kit.app
        import omni.usd
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        usd_output_dir.mkdir(parents=True, exist_ok=True)

        config = URDFImporterConfig()
        config.urdf_path = str(urdf_output)
        config.usd_path = str(usd_output_dir)
        config.make_default_prim = True
        config.fix_base = True
        config.merge_mesh = False
        config.collision_from_visuals = False
        config.self_collision = False
        config.import_inertia_tensor = True
        _set_if_present(config, "parse_mimic", True)
        _set_if_present(config, "import_mimic", True)
        _set_if_present(config, "default_drive_type", "position")

        importer = URDFImporter(config)
        usd_path = Path(importer.import_urdf()).resolve()
        print(f"[INFO] imported USD: {usd_path}")

        omni.usd.get_context().open_stage(str(usd_path))
        app = omni.kit.app.get_app()
        for _ in range(20):
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open generated USD: {usd_path}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError(f"Generated USD has no default prim: {usd_path}")
        print(f"[INFO] defaultPrim: {default_prim.GetPath()}")

        stage.GetRootLayer().Save()
        _patch_articulation_root_for_isaac_lab(usd_path, default_prim.GetName())
        print(f"[INFO] saved patched USD: {usd_path}")
        _create_robotiq_control_copy(usd_path, default_prim.GetName(), "physx")
        _create_robotiq_control_copy(usd_path, default_prim.GetName(), "newton")

        common_asset_dir = repo_root / "isaaclab_arena/assets/robots/abb/irb1200_7_70"
        mesh_src = abb_root / "abb_irb1200_support/meshes"
        mesh_dst = common_asset_dir / "meshes"
        if mesh_src.exists() and not mesh_dst.exists():
            shutil.copytree(mesh_src, mesh_dst)
            print(f"[INFO] copied meshes: {mesh_dst}")

        print(usd_path)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
