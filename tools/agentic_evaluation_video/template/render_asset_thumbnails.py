# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render isolated USD assets and use renderer depth for transparent PNG backgrounds."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="*")
args = parser.parse_args()
from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True, "enable_cameras": True}).app

# Load the USD-only helper without importing the embodiments package's simulation modules.
import importlib.util
import numpy as np
import sys

import carb
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

import isaaclab_arena
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR, ISAAC_STAGING_NUCLEUS_DIR

helper_path = Path(isaaclab_arena.__file__).resolve().parent / "embodiments/robot_on_stand_utils.py"
helper_spec = importlib.util.spec_from_file_location("thumbnail_robot_stand", helper_path)
stand_helper = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = stand_helper
helper_spec.loader.exec_module(stand_helper)
RobotPrimSpec = stand_helper.RobotPrimSpec
StandPrimSpec = stand_helper.StandPrimSpec
compose_on_stand_usd = stand_helper.compose_on_stand_usd
from isaaclab_arena.utils.usd.articulation import (
    articulation_joint_prims,
    compute_posed_prim_world_deltas,
    resolve_joint_pos_patterns,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "rendered"
OUT.mkdir(parents=True, exist_ok=True)
# Match the source YAML's registered USDs, scale and 0.8 m DROID stand.
robot_usd = compose_on_stand_usd(
    RobotPrimSpec(
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
        "/panda",
        "panda_link0",
        "stand_instanceable",
    ),
    StandPrimSpec(
        f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/robots/franka_stand_grey.usda",
        "/World/franka_table",
        "franka_table",
        (-0.05, 0.0, 0.0),
        (1.08, 0.91032),
        1.35,
    ),
    stand_height_m=0.8,
    output_basename="droid_franka_robotiq_on_stand",
)
object_root = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects"
kitchen_usd = f"{ISAAC_STAGING_NUCLEUS_DIR}/Environments/replicator_kitchen/kitchen_peninsula.usda"
specs = {
    "banana": (f"{object_root}/ycb/banana.usd", None, (1, 1, 1)),
    "orange": (f"{object_root}/fruits_veggies/orange_01.usd", None, (1, 1, 1)),
    "near": (f"{object_root}/hot3d/wooden_bowl.usd", None, (1, 1, 1)),
    "far": (f"{object_root}/ycb/bowl.usd", None, (1.25, 1.25, 1)),
    "robot": (robot_usd, None, (1, 1, 1)),
    "kitchen": (kitchen_usd, None, (1, 1, 1)),
    "island": (kitchen_usd, "CapSurface_02", (1, 1, 1)),
    "floor": (kitchen_usd, "KitchenRoom/Floor", (1, 1, 1)),
}
rep.orchestrator.set_capture_on_play(False)
carb.settings.get_settings().set("/rtx/post/aa/op", 0)
records = json.loads((OUT / "sources.json").read_text()) if (OUT / "sources.json").exists() else {}
try:
    for key, (usd_path, reference, scale) in specs.items():
        if args.only and key not in args.only:
            continue
        print(f"RENDER_START {key} {usd_path}", flush=True)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = stage.DefinePrim("/Asset", "Xform")
        root.GetReferences().AddReference(usd_path)
        # Studio lighting replaces embedded lights, whose bounds can dwarf the room.
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdLux.LightAPI):
                prim.SetActive(False)
        hidden = []
        if key == "kitchen":
            # A cutaway reveals the interior when the complete room is viewed from outside.
            hidden = ["KitchenRoom/Ceiling", "KitchenRoom/Wall_East", "KitchenRoom/Wall_South"]
            for relative_path in hidden:
                UsdGeom.Imageable(stage.GetPrimAtPath("/Asset/" + relative_path)).MakeInvisible()
        # Hide all other kitchen geometry while keeping materials and parent transforms.
        target = root
        if reference:
            target = stage.GetPrimAtPath("/Asset/" + reference)
            assert target, reference
            for prim in stage.Traverse():
                p = prim.GetPath()
                if p.HasPrefix(target.GetPath()) or target.GetPath().HasPrefix(p):
                    continue
                if p.HasPrefix(root.GetPath()) and prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
        UsdGeom.Xformable(root).AddScaleOp(opSuffix="thumbnail").Set(Gf.Vec3f(*scale))
        if key == "robot":
            joints = resolve_joint_pos_patterns(
                articulation_joint_prims(root),
                {
                    "panda_joint1": 0.0,
                    "panda_joint2": -np.pi / 5,
                    "panda_joint3": 0.0,
                    "panda_joint4": -4 * np.pi / 5,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3 * np.pi / 5,
                    "panda_joint7": 0.0,
                    "finger_joint": 0.0,
                },
            )
            deltas = compute_posed_prim_world_deltas(stage, "/Asset", joints)
            cache = UsdGeom.XformCache()
            transforms = {
                p: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) * Gf.Matrix4d(delta.tolist())
                for p, delta in deltas.items()
            }
            for p, matrix in transforms.items():
                xf = UsdGeom.Xformable(stage.GetPrimAtPath(p))
                xf.ClearXformOpOrder()
                xf.AddTransformOp(opSuffix="thumbnailPose").Set(matrix)
                xf.SetResetXformStack(True)
        for _ in range(12):
            app.update()
        bounds = (
            UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False
            )
            .ComputeWorldBound(target)
            .ComputeAlignedRange()
        )
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        center = (low + high) * 0.5
        extent = high - low
        assert np.all(np.isfinite(extent)) and np.max(extent) > 0, (key, low, high)
        # Identical elevated studio angle for small props; enough clearance to see bowl interiors.
        direction = np.array((1.1, 1.3, 1.15))
        if key == "kitchen":
            direction = np.array((1.1, -1.3, 1.6))
        if key == "robot":
            direction = np.array((1.3, 1.5, 0.8))
        direction /= np.linalg.norm(direction)
        distance = np.linalg.norm(extent) * 3.0
        eye = center + direction * distance
        camera = rep.create.camera(
            position=tuple(eye), look_at=tuple(center), focal_length=50, clipping_range=(0.001, 10000)
        )
        for camera_path in camera.get_output_prims()["prims"]:
            (
                camera_path if isinstance(camera_path, Usd.Prim) else stage.GetPrimAtPath(str(camera_path))
            ).CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(0.0)
        render_product = rep.create.render_product(camera, (1024, 1024))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        rgb.attach([render_product])
        depth.attach([render_product])
        dome = UsdLux.DomeLight.Define(stage, "/StudioDome")
        dome.CreateIntensityAttr(500)
        # Broad neutral lighting keeps the orange and the two bowl materials legible at small sizes.
        light = UsdLux.DistantLight.Define(stage, "/StudioKey")
        light.CreateIntensityAttr(800)
        light.CreateAngleAttr(20)
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -25))
        # App updates pump the headless render products without advancing physics.
        for _ in range(180):
            app.update()
        color = np.asarray(rgb.get_data()).copy()
        distances = np.asarray(depth.get_data()).squeeze().copy()
        assert color.shape[:2] == (1024, 1024), color.shape
        foreground = np.isfinite(distances) & (distances > 0) & (distances < distance * 10)
        assert foreground.any() and not foreground.all(), (key, float(foreground.mean()))
        rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
        rgba[:, :, :3] = color[:, :, :3]
        rgba[:, :, 3] = foreground.astype(np.uint8) * 255
        image = Image.fromarray(rgba)
        bbox = image.getbbox()
        assert bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 1024 and bbox[3] < 1024, (key, bbox)
        image = image.crop(bbox)
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        output = Image.new("RGBA", (512, 512))
        output.paste(image, ((512 - image.width) // 2, (512 - image.height) // 2))
        output.save(OUT / f"{key}.png")
        records[key] = {
            "usd_path": usd_path,
            "reference": reference,
            "scale": scale,
            "eye": eye.tolist(),
            "target": center.tolist(),
            "bounds": [low.tolist(), high.tolist()],
            "hidden_for_cutaway": hidden,
            "studio_lighting": {"dome_intensity": 500, "key_intensity": 800},
            "alpha": "renderer distance_to_camera foreground mask",
            "size": [512, 512],
        }
        (OUT / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
        rgb.detach([render_product])
        depth.detach([render_product])
        render_product.destroy()
        print(f"RENDER_DONE {key} bbox={bbox}", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
