#!/usr/bin/env python3
"""Assemble an ABB IRB1200 + Robotiq 2F-85 (Isaac Collected) USD for Newton.

The 2F-85 source is the Isaac 6.0 Collected asset under
``examples/examples_teleop/Collected_2F-85``. Its ``physics_mimic`` variant
(6 revolute DOF: finger_joint + 5 followers via PhysxMimicJointAPI) is referenced
into the ABB link_6 with a fixed joint. The mimic joint multipliers are enforced
in software in the example (Newton does not reliably solve mimic under contact).

Run inside the IsaacLab-Arena Docker workspace:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh tools/convert_2f85_to_abb_irb1200_usd.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def write_abb_2f85_combined_usd(
    abb_source_dir: Path,
    f85_source_dir: Path,
    combined_dir: Path,
    mount_translate: tuple[float, float, float],
    mount_orient: tuple[float, float, float, float],
) -> Path:
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    shutil.copytree(abb_source_dir, combined_dir)

    # Copy the 2F-85 Collected asset (payloads/parts/materials) preserving relative refs.
    f85_dst = combined_dir / "robotiq_2f85"
    for sub in ("payloads", "parts", "materials"):
        src = f85_source_dir / sub
        if src.exists():
            shutil.copytree(src, f85_dst / sub)

    # Strip the articulation root off the inner Robotiq_2F_85 prim so it can be
    # welded to the ABB link_6 as a plain sub-articulation (otherwise Newton sees
    # a cycle: the floating articulation root + our fixed joint).
    mimic_usd = f85_dst / "payloads" / "Robotiq_2F_85_phyisics_mimic.usda"
    mt = mimic_usd.read_text(encoding="utf-8")
    mt = mt.replace(
        'prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]',
        '',
    )
    mimic_usd.write_text(mt, encoding="utf-8")

    combined_usd_path = combined_dir / "irb1200_7_70.usda"
    default_prim_name = "abb_irb1200_7_70"
    rel_f85 = "robotiq_2f85/payloads/Robotiq_2F_85_phyisics_mimic.usda"
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
                                                def Xform "robotiq_2f85_mount" (
                                                    prepend references = @{rel_f85}@
                                                )
                                                {{
                                                    quatd xformOp:orient = {mount_orient}
                                                    double3 xformOp:translate = {mount_translate}
                                                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

                                                    over "Robotiq_2F_85"
                                                    {{
                                                        over "Robotiq_2F_85" (
                                                            delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI"]
                                                        )
                                                        {{
                                                        }}
                                                    }}
                                                }}

                                                def PhysicsFixedJoint "robotiq_2f85_fixed_joint"
                                                {{
                                                    rel physics:body0 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6>
                                                    rel physics:body1 = </{default_prim_name}/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6/robotiq_2f85_mount/Robotiq_2F_85/Robotiq_2F_85/base_link>
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
        "--abb-source-dir",
        type=Path,
        default=Path("isaaclab_arena/assets/robots/abb/irb1200_7_70"),
        help="Existing converted ABB IRB1200 USD directory to copy.",
    )
    parser.add_argument(
        "--f85-source-dir",
        type=Path,
        default=Path("examples/examples_teleop/Collected_2F-85"),
        help="Isaac Collected Robotiq 2F-85 asset directory.",
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=Path("examples/examples_teleop/assets/abb_irb1200_robotiq_2f85_collected_newton"),
        help="Output directory for the assembled ABB + 2F-85 USD.",
    )
    # mount rpy=(pi, 0, 0): flip the fingers (default +Z) to point -Z (down).
    parser.add_argument("--mount-translate", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--mount-rpy", type=float, nargs=3, default=(3.141592653589793, 0.0, 0.0))
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    abb_source_dir = (repo_root / args.abb_source_dir).resolve()
    f85_source_dir = (repo_root / args.f85_source_dir).resolve()
    combined_dir = (repo_root / args.combined_dir).resolve()

    # rpy=(roll,pitch,yaw) -> quat about X then Y then Z (roll=pi -> rotate X 180deg)
    import numpy as np

    roll, pitch, yaw = args.mount_rpy
    qx = np.array([np.sin(roll / 2), 0, 0, np.cos(roll / 2)])
    qy = np.array([0, np.sin(pitch / 2), 0, np.cos(pitch / 2)])
    qz = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
    # quaternion multiply qz * qy * qx (Hamilton product, [x,y,z,w])
    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    orient = qmul(qz, qmul(qy, qx))
    orient = tuple(float(v) for v in orient)

    out = write_abb_2f85_combined_usd(
        abb_source_dir,
        f85_source_dir,
        combined_dir,
        tuple(args.mount_translate),
        orient,
    )
    print(f"[INFO] wrote ABB + 2F-85 USD: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
