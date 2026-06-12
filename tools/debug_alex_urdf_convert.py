"""Minimal URDF-to-USD conversion test for the Alex robot.

Run inside the container to capture urdf_usd_converter errors:

    /isaac-sim/python.sh tools/debug_alex_urdf_convert.py
    /isaac-sim/python.sh tools/debug_alex_urdf_convert.py --robot-version V2
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot-version", choices=["V1", "V2"], default="V1")
parser.add_argument("--models_dir", default="/models")
args = parser.parse_args()

import math
import os
import tempfile
import xml.etree.ElementTree as ET

from isaaclab_arena.embodiments.alex.alex import (
    ALEX_NUBFOREARMS_PARTS,
    _alex_arena_urdf_paths,
    _mesh_path_replacements,
    merge_urdfs,
)

paths = _alex_arena_urdf_paths(args.robot_version)
ver = args.robot_version.lower()
merged_urdf = merge_urdfs(
    args.robot_version,
    ALEX_NUBFOREARMS_PARTS,
    output_name=f"alex_{ver}_nubs_arena",
)

# ── Step 1: build the normalized URDF (axis fix + package:// resolve) ──────
normalized_path = "/tmp/alex_debug_normalized.urdf"
tree = ET.parse(merged_urdf)
root = tree.getroot()
replacements = _mesh_path_replacements()
for axis_el in root.iter("axis"):
    parts = [float(v) for v in axis_el.get("xyz", "0 0 0").split()]
    length = math.sqrt(sum(v * v for v in parts))
    if length > 1e-9 and abs(length - 1.0) > 1e-6:
        axis_el.set("xyz", " ".join(f"{v / length:.6f}" for v in parts))
for mesh_el in root.iter("mesh"):
    fn = mesh_el.get("filename", "")
    for pkg_prefix, abs_prefix in replacements.items():
        if fn.startswith(pkg_prefix):
            mesh_el.set("filename", abs_prefix + fn[len(pkg_prefix):])
            break
tree.write(normalized_path, xml_declaration=True, encoding="unicode")
print(f"[1] Normalized URDF written to {normalized_path}")

# ── Step 2: merge fixed joints (same logic as UrdfConverter) ──────────────
import sys

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab/isaaclab/sim/converters")
from urdf_utils import merge_fixed_joints

fd, merged_path = tempfile.mkstemp(suffix=".urdf", prefix=".merged_alex_", dir="/tmp")
os.close(fd)
merge_fixed_joints(normalized_path, merged_path)
print(f"[2] Merged URDF written to {merged_path}")

# ── Step 3: Isaac Sim bootstrap ───────────────────────────────────────────
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("[3] SimulationApp started")

import carb
import importlib

try:
    import gc
    import shutil
    from isaacsim.asset.importer.utils.impl import importer_utils, stage_utils

    # ── Step 4: urdf_usd_converter ─────────────────────────────────────────
    usd_dir = f"/tmp/alex_debug_full_{ver}"
    usdex_path = os.path.join(usd_dir, "usdex")
    intermediate_path = os.path.join(usd_dir, "temp", f"alex_{ver}_normalized.usd")
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)

    urdf_usd_converter = importlib.import_module("urdf_usd_converter")
    converter = urdf_usd_converter.Converter(layer_structure=False, scene=False)
    print(f"[4] convert({merged_path!r}, {usdex_path!r})")
    asset = converter.convert(merged_path, usdex_path)
    print(f"[4] → {asset!r}, exists={os.path.exists(asset.path) if asset else False}")

    # ── Step 5: open intermediate stage ───────────────────────────────────
    print(f"[5] open_stage({asset.path!r})")
    stage = stage_utils.open_stage(asset.path)
    print(f"[5] → stage={stage!r}")
    if not stage:
        raise ValueError(f"Failed to open intermediate stage: {asset.path}")

    importer_utils.remove_custom_scopes(stage)
    importer_utils.add_rigid_body_schemas(stage)
    importer_utils.add_joint_schemas(stage)
    importer_utils.enable_self_collision(stage, False)
    print("[5] post-processing done")

    # ── Step 6: save intermediate stage ───────────────────────────────────
    print(f"[6] save_stage({intermediate_path!r})")
    stage_utils.save_stage(stage, intermediate_path)
    stage = None
    gc.collect()
    print(f"[6] → exists={os.path.exists(intermediate_path)}")

    # ── Step 7: asset transformer ──────────────────────────────────────────
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.transformer.rules")
    ext_path = ext_manager.get_extension_path(ext_id)
    profile_json = os.path.normpath(os.path.join(ext_path, "data", "isaacsim_structure.json"))
    output_root = os.path.join(usd_dir, f"alex_{ver}_normalized")
    print(f"[7] run_asset_transformer_profile(input={intermediate_path!r}, output_root={output_root!r})")
    importer_utils.run_asset_transformer_profile(
        input_stage_path=intermediate_path,
        output_package_root=output_root,
        profile_json_path=profile_json,
    )
    final_usd = os.path.join(output_root, f"alex_{ver}_normalized.usda")
    print(f"[7] → final USD exists: {os.path.exists(final_usd)}")
    if not os.path.exists(final_usd):
        print(f"[7] Contents of {output_root}:")
        if os.path.isdir(output_root):
            for f in os.listdir(output_root):
                print(f"      {f}")
        else:
            print("      (directory does not exist)")

except Exception as e:
    import traceback

    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    app.close()
