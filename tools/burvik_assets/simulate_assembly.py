# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a contact-active BURVIK assembly and measure released-body stability in Isaac Sim."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--output", default="results")
args = parser.parse_args()
ROOT = Path(os.environ.get("BURVIK_ASSET_DIR", Path(__file__).resolve().parents[2] / "artifacts/ikea_burvik"))
OUT = ROOT / args.output
OUT.mkdir(parents=True, exist_ok=True)

from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

launcher = get_app_launcher(
    argparse.Namespace(
        headless=True,
        enable_cameras=args.video or args.smoke,
        visualizer=[],
        livestream=-1,
        width=1280,
        height=720,
    )
)
app = launcher.app

# The minimal Arena experience deliberately does not enable the legacy Core extensions.
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.add_path("/isaac-sim/extsDeprecated")
assert manager.set_extension_enabled_immediate("isaacsim.core.api", True)
if args.video or args.smoke:
    assert manager.set_extension_enabled_immediate("isaacsim.sensors.camera", True)

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import carb
import imageio.v2 as imageio
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

if args.video or args.smoke:
    from isaacsim.sensors.camera import Camera

DT = 1 / 240
FPS = 24
BENCH_Z = 0.75
SPEC = json.loads((ROOT / "specification.json").read_text())
NAMES = ["tray", "leg_1", "leg_2", "leg_3", "leg_4", "handle"]
world = World(
    physics_dt=DT,
    rendering_dt=1 / FPS,
    stage_units_in_meters=1,
    backend="numpy",
    device="cpu",
    sim_params={
        "use_gpu_pipeline": False,
        "use_fabric": False,
        "enable_scene_query_support": True,
    },
)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1)
stage.SetTimeCodesPerSecond(FPS)
world_root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
stage.SetDefaultPrim(world_root)
settings = carb.settings.get_settings()
settings.set("/physics/updateToUsd", True)
settings.set("/physics/updateVelocitiesToUsd", True)
settings.set("/rtx/post/dlss/execMode", 2)


def log(message):
    print("[BURVIK] " + message, flush=True)


def quat(rotation):
    xyzw = rotation.as_quat()
    return np.array([xyzw[3], *xyzw[:3]], dtype=np.float32)


def rotation(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def material(path, color, roughness=0.45):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


bench_mat = material("/World/Materials/Bench", (0.14, 0.19, 0.22))
floor_mat = material("/World/Materials/Floor", (0.055, 0.072, 0.094))
frame_mat = material("/World/Materials/Frame", (0.055, 0.07, 0.083), 0.35)


def box(path, pos, size, mat, collision=True):
    geo = UsdGeom.Cube.Define(stage, path)
    geo.CreateSizeAttr(1)
    geo.AddTranslateOp().Set(Gf.Vec3d(*pos))
    geo.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        api = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
        api.CreateContactOffsetAttr(0.00015)
        api.CreateRestOffsetAttr(0)
    return geo


box("/World/Bench/Top", (0, 0, BENCH_Z - 0.035), (2.0, 1.5, 0.07), bench_mat)
for x in (-0.82, 0.82):
    for y in (-0.58, 0.58):
        box(
            f"/World/Bench/Support_{x}_{y}".replace("-", "n").replace(".", "_"),
            (x, y, 0.35),
            (0.06, 0.06, 0.70),
            frame_mat,
        )
box("/World/Floor", (0, 0, -0.035), (200, 200, 0.05), floor_mat)
light = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
light.CreateIntensityAttr(400)
light.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1))
for name, pos, intensity, color, scale in [
    ("Key", (1, -1.5, 3.5), 2300, (1, 0.92, 0.82), 2),
    ("Fill", (-1.5, -0.5, 2.7), 1600, (0.83, 0.9, 1), 1.5),
    ("Rim", (0, 1.5, 3), 1800, (1, 1, 1), 1.5),
]:
    rect_light = UsdLux.RectLight.Define(stage, "/World/Lighting/" + name)
    rect_light.CreateIntensityAttr(intensity)
    rect_light.CreateWidthAttr(scale)
    rect_light.CreateHeightAttr(scale)
    rect_light.CreateColorAttr(Gf.Vec3f(*color))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 0, 1)).GetInverse()
    rect_light.AddTransformOp().Set(view)

initial = {
    "tray": (
        np.array([0, 0.07, BENCH_Z + 0.064]),
        Rotation.from_euler("x", 180, degrees=True),
    ),
    "handle": (
        np.array([0, -0.30, BENCH_Z + 0.008]),
        Rotation.from_euler("x", 90, degrees=True),
    ),
}
for i, x in enumerate([-0.60, -0.39, 0.39, 0.60]):
    initial[f"leg_{i + 1}"] = (
        np.array([x, -0.15, BENCH_Z + 0.012]),
        Rotation.from_euler("x", 90, degrees=True),
    )

bodies = {}
for name in NAMES:
    prim = UsdGeom.Xform.Define(stage, "/World/Parts/" + name).GetPrim()
    prim.GetReferences().AddReference(str(ROOT / "assets" / f"{name}.usd"))
    pos, rot = initial[name]
    body = RigidPrim(
        "/World/Parts/" + name,
        name=name,
        positions=np.array([pos], dtype=np.float32),
        orientations=np.array([quat(rot)]),
        track_contact_forces=True,
    )
    bodies[name] = world.scene.add(body)

scene_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
scene_api.CreateEnableCCDAttr(True)
scene_api.CreateEnableGPUDynamicsAttr(False)
scene_api.CreateBroadphaseTypeAttr("MBP")
scene_api.CreateSolverTypeAttr("TGS")

camera = None
writer = None
if args.video or args.smoke:
    camera = Camera("/World/Camera", resolution=(1280, 720), frequency=FPS)
    camera.set_focal_length(35)
    camera.set_horizontal_aperture(36)
    camera.set_clipping_range(0.02, 100)

world.reset()
if camera:
    camera.initialize()


def set_camera(eye=(1.35, -1.95, 2.20), target=(0, 0, 1.20)):
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    q = matrix.ExtractRotationQuat()
    camera.set_world_pose(np.array(eye), np.array([q.GetReal(), *q.GetImaginary()]), camera_axes="usd")


if camera:
    set_camera()
    for _ in range(20):
        world.render()
if args.video:
    writer = imageio.get_writer(
        str(OUT / "burvik_assembly.mp4"),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except OSError:
    font = ImageFont.load_default()
    font_small = font

report = {
    "runtime": "Isaac Sim / PhysX, isaaclab_arena:latest",
    "dt_s": DT,
    "masses_kg": SPEC["masses_kg"],
    "insertion_events": [],
    "stability": {},
    "phases": [],
    "retention": "Fixed part-to-part joint activated only at seated pose; no world joints.",
    "control": "Contact-active dynamic rigid bodies with bounded scripted velocity/displacement guidance.",
}
trajectory = []
step_count = 0
phase = "Loose parts | gravity settling"
detail = "Six independent dynamic bodies on the workbench"
active_drives = {}
attached = []
last_wall = time.monotonic()


def pose(name):
    p, q = bodies[name].get_world_poses()
    return np.array(p[0], dtype=float), rotation(q[0])


def phase_set(title, subtitle):
    global phase, detail
    phase, detail = title, subtitle
    report["phases"].append({"time_s": step_count * DT, "title": title, "detail": subtitle})
    log(title)


def drive(name, target, target_rotation, max_speed=0.3):
    p, r = pose(name)
    v = 18 * (np.array(target) - p)
    speed = np.linalg.norm(v)
    if speed > max_speed:
        v *= max_speed / speed
    v[2] += 9.81 * DT
    w = 16 * (target_rotation * r.inv()).as_rotvec()
    angular_speed = np.linalg.norm(w)
    if angular_speed > 1.5:
        w *= 1.5 / angular_speed
    bodies[name].set_linear_velocities(np.array([v], dtype=np.float32))
    bodies[name].set_angular_velocities(np.array([w], dtype=np.float32))


def capture():
    rgb = camera.get_rgb()
    if rgb is None or rgb.size == 0:
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    picture = Image.fromarray(rgb[:, :, :3])
    draw = ImageDraw.Draw(picture, "RGBA")
    draw.rectangle((0, 0, 1280, 104), fill=(13, 20, 30, 230))
    draw.text((30, 16), "BURVIK / PHYSICS ASSEMBLY", font=font, fill=(240, 230, 209))
    draw.text((30, 52), phase, font=font, fill=(255, 255, 255))
    draw.rectangle((0, 664, 1280, 720), fill=(13, 20, 30, 225))
    draw.text((30, 681), detail, font=font_small, fill=(225, 230, 234))
    draw.text((1080, 20), f"{step_count * DT:05.1f} s", font=font, fill=(164, 215, 227))
    if writer:
        writer.append_data(np.asarray(picture))
    if step_count % 240 == 0:
        picture.save(OUT / "latest_frame.png")
    return picture


def step(duration, extra=None):
    global step_count, last_wall
    for _ in range(round(duration / DT)):
        for name, (p, r, speed) in active_drives.items():
            drive(name, p, r, speed)
        if extra:
            extra()
        world.step(render=False)
        step_count += 1
        if step_count % 10 == 0:
            frame = {
                "time_s": step_count * DT,
                "phase": phase,
                "driven": list(active_drives),
                "poses": {},
            }
            for name in NAMES:
                p, r = pose(name)
                frame["poses"][name] = [*p.tolist(), *quat(r).tolist()]
            trajectory.append(frame)
            if camera:
                world.render()
                capture()
        if time.monotonic() - last_wall > 20:
            log(f"t={step_count * DT:.1f}s | {phase}")
            last_wall = time.monotonic()


def move(name, target, rot, duration, speed=0.3):
    start_p, start_r = pose(name)
    duration = max(
        duration,
        1.7 * np.linalg.norm(np.array(target) - start_p) / speed,
        1.7 * np.linalg.norm((rot * start_r.inv()).as_rotvec()) / 1.5,
    )
    slerp = Slerp([0, 1], Rotation.concatenate([start_r, rot]))
    for tick in range(round(duration / DT)):
        t = (tick + 1) / (duration / DT)
        s = t * t * (3 - 2 * t)
        active_drives[name] = (
            start_p + s * (np.array(target) - start_p),
            slerp([s])[0],
            speed,
        )
        step(DT)
    active_drives[name] = (np.array(target), rot, speed)
    step(0.25)
    for _ in range(240):
        current_p, current_r = pose(name)
        if np.linalg.norm(current_p - target) < 0.001 and np.linalg.norm((rot * current_r.inv()).as_rotvec()) < 0.008:
            break
        step(DT)


def sample_stability(label, duration):
    samples = {name: [] for name in NAMES}
    velocities = {name: [] for name in NAMES}
    contacts = {name: [] for name in NAMES}
    assert not active_drives, "Stability checks must have no scripted supports."
    for _ in range(round(duration / 0.05)):
        step(0.05)
        for name in NAMES:
            p, r = pose(name)
            samples[name].append([*p, *quat(r)])
            velocities[name].append(bodies[name].get_velocities()[0].tolist())
            contacts[name].append(bodies[name].get_net_contact_forces(dt=DT)[0].tolist())
    output = {}
    for name in NAMES:
        values = np.array(samples[name])
        speeds = np.array(velocities[name])
        ref = rotation(values[0, 3:])
        angle = max(np.linalg.norm((rotation(row[3:]) * ref.inv()).as_rotvec()) for row in values)
        output[name] = {
            "max_position_drift_m": float(np.max(np.linalg.norm(values[:, :3] - values[0, :3], axis=1))),
            "max_rotation_drift_deg": float(np.degrees(angle)),
            "max_linear_speed_m_s": float(np.max(np.linalg.norm(speeds[:, :3], axis=1))),
            "max_angular_speed_rad_s": float(np.max(np.linalg.norm(speeds[:, 3:], axis=1))),
            "mean_contact_force_N": np.mean(contacts[name], axis=0).tolist(),
            "final_pose": values[-1].tolist(),
        }
    report["stability"][label] = {
        "duration_s": duration,
        "drives_active": False,
        "parts": output,
        "passed": all(
            v["max_position_drift_m"] < 0.002 and v["max_rotation_drift_deg"] < 1 and v["max_linear_speed_m_s"] < 0.02
            for v in output.values()
        ),
    }
    log(f'{label}: {json.dumps(report["stability"][label])}')


def socket_target(index):
    p, r = pose("tray")
    socket = SPEC["sockets"][index]
    seat = p + r.apply(socket["seat"])
    axis = r.apply(socket["axis"])
    local = Rotation.align_vectors([socket["axis"]], [[0, 0, 1]])[0]
    return seat, r * local, axis


def latch(name, ideal_p, ideal_r, insertion):
    p, r = pose(name)
    distance = np.linalg.norm(p - ideal_p)
    angle = np.degrees(np.linalg.norm((r * ideal_r.inv()).as_rotvec()))
    log(
        f"Seating check {name}: delta={(p - ideal_p).tolist()}, angle={angle},"
        f" contact={bodies[name].get_net_contact_forces(dt=DT).tolist()}"
    )
    assert distance < 0.0012 and angle < 1, f"{name} not seated: {distance * 1000:.2f} mm, {angle:.2f} deg"
    tray_p, tray_r = pose("tray")
    relative_p = tray_r.inv().apply(p - tray_p)
    relative_r = tray_r.inv() * r
    j = UsdPhysics.FixedJoint.Define(stage, "/World/Connections/" + name)
    j.CreateBody0Rel().SetTargets(["/World/Parts/tray"])
    j.CreateBody1Rel().SetTargets(["/World/Parts/" + name])
    j.CreateLocalPos0Attr(Gf.Vec3f(*relative_p))
    j.CreateLocalRot0Attr(Gf.Quatf(float(quat(relative_r)[0]), Gf.Vec3f(*quat(relative_r)[1:].tolist())))
    j.CreateLocalPos1Attr(Gf.Vec3f(0))
    j.CreateLocalRot1Attr(Gf.Quatf(1))
    j.CreateCollisionEnabledAttr(False)
    j.CreateBreakForceAttr(500)
    j.CreateBreakTorqueAttr(80)
    attached.append(name)
    active_drives.pop(name, None)
    report["insertion_events"].append({
        "part": name,
        "time_s": step_count * DT,
        "seating_error_m": float(distance),
        "angular_error_deg": float(angle),
        "nominal_insertion_m": insertion,
        "joint": str(j.GetPath()),
    })
    step(0.35)
    log(f"Seated {name}: {distance * 1000:.3f} mm, {angle:.3f} deg")


def insert_leg(index):
    name = f"leg_{index + 1}"
    if camera:
        set_camera()
    phase_set(
        f"Insert leg {index + 1} / 4",
        "Smooth 8 mm pin | open socket | collisions active until seated",
    )
    p, r = pose(name)
    # Pick vertically clear of the bench before rotating the long part.
    move(name, p + np.array([0, 0, 0.32]), r, 1.0)
    seat, goal_r, axis = socket_target(index)
    pre = seat - axis * 0.085
    if abs(axis[0]) > 0.8:
        move(name, (p[0], p[1], 1.90), r, 1.4)
        clearance = np.array([pre[0], pre[1], 1.90])
    else:
        clearance = np.array([pre[0], pre[1], max(pre[2] + 0.16, 1.24)])
    move(name, clearance, goal_r, 1.6)
    move(name, pre, goal_r, 1.2)
    if camera:
        eye = seat + (np.array([-0.38, -0.43, 0.23]) if abs(axis[0]) > 0.8 else np.array([0.35, -0.42, 0.30]))
        set_camera(eye, seat + np.array([0, 0, 0.025]))
    # Recompute against the tray's measured pose during the final insertion.
    for tick in range(600):
        seat, goal_r, axis = socket_target(index)
        distance = 0.085 * (1 - (tick + 1) / 600)
        active_drives[name] = (seat - axis * distance, goal_r, 0.045)
        step(DT)
    for _ in range(240):
        seat, goal_r, axis = socket_target(index)
        active_drives[name] = (seat, goal_r, 0.035)
        step(DT)
    latch(name, seat, goal_r, 0.018)
    if camera:
        set_camera()


def insert_handle():
    phase_set(
        "Insert carry handle",
        "Both smooth handle pins enter the two opposing tray holes together",
    )
    p, r = pose("handle")
    move("handle", p + np.array([0, 0, 0.32]), r, 1.0)
    tray_p, tray_r = pose("tray")
    axis = tray_r.apply([0, 0, 1])
    pre = tray_p + axis * 0.09
    move("handle", pre + np.array([0, 0, 0.26]), tray_r, 1.5)
    move("handle", pre, tray_r, 1.2)
    if camera:
        set_camera(tray_p + np.array([0.57, -0.57, 0.32]), tray_p)
    for tick in range(480):
        tray_p, tray_r = pose("tray")
        axis = tray_r.apply([0, 0, 1])
        active_drives["handle"] = (
            tray_p + axis * 0.09 * (1 - (tick + 1) / 480),
            tray_r,
            0.05,
        )
        step(DT)
    for _ in range(240):
        tray_p, tray_r = pose("tray")
        active_drives["handle"] = (tray_p, tray_r, 0.02)
        step(DT)
    latch("handle", tray_p, tray_r, 0.014)
    if camera:
        set_camera()


def save_scene(filename):
    # Edit a separate layer so rewriting references cannot invalidate live PhysX views.
    snapshots = {name: pose(name) for name in NAMES}
    output_layer = Sdf.Layer.CreateNew(str(OUT / filename))
    output_layer.TransferContent(stage.GetRootLayer())
    saved_stage = Usd.Stage.Open(output_layer)
    for name in NAMES:
        p, r = snapshots[name]
        xf = UsdGeom.Xformable(saved_stage.GetPrimAtPath("/World/Parts/" + name))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*p))
        q = quat(r)
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:].tolist())))
        prim = xf.GetPrim()
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference("../assets/" + name + ".usd")
    output_layer.Save()


try:
    log("Begin gravity settling")
    step(2)
    sample_stability("loose_parts", 2)
    if camera:
        picture = capture()
        if picture:
            picture.save(OUT / "loose_parts.png")
    if args.smoke:
        (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2))
        log("SMOKE COMPLETE")
    else:
        assert report["stability"]["loose_parts"]["passed"], "Loose parts are not settled."
        tray_p, tray_r = pose("tray")
        active_drives["tray"] = (tray_p, tray_r, 0.25)
        insert_leg(1)
        insert_leg(3)
        phase_set(
            "Reorient for the handle",
            "Scripted handling of the tray; attached legs remain dynamic through joints",
        )
        move("tray", (0, 0.06, 1.42), tray_r, 2.0)
        side = Rotation.from_euler("y", 90, degrees=True)
        move("tray", (0, 0.06, 1.42), side, 2.5)
        insert_handle()
        insert_leg(0)
        insert_leg(2)
        phase_set(
            "Turn upright and lower",
            "All six parts joined | only the tray receives scripted guidance",
        )
        move("tray", (0, 0.06, 1.42), Rotation.identity(), 2.5)
        move("tray", (0, 0.06, BENCH_Z + 0.424), Rotation.identity(), 2.0, 0.1)
        active_drives.clear()
        phase_set(
            "Release / free-standing stability",
            "All controls OFF | gravity + contacts + five part-to-part retention joints",
        )
        step(2)
        sample_stability("assembled_free", 5)
        # A small external lateral impulse checks recovery without anchoring the table.
        phase_set(
            "Lateral disturbance / recovery",
            "1 N lateral force for 0.25 s, then all forces removed",
        )
        step(
            0.25,
            extra=lambda: bodies["tray"].apply_forces(np.array([[1.0, 0, 0]], dtype=np.float32)),
        )
        step(2)
        sample_stability("after_disturbance", 3)
        if camera:
            phase_set(
                "Assembly complete",
                "38 cm tray | 78 cm overall height | released and physically stable",
            )
            set_camera((1.1, -1.55, 2.0), (0, 0.06, 1.18))
            step(2)
            picture = capture()
            if picture:
                picture.save(OUT / "assembled.png")
        report["passed"] = all(v["passed"] for v in report["stability"].values()) and len(attached) == 5
        report["world_fixed_joints"] = 0
        report["all_bodies_dynamic"] = all(
            not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/Parts/" + n)).GetKinematicEnabledAttr().Get()
            for n in NAMES
        )
        (OUT / "validation.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(OUT / "trajectory.npz", frames=np.array(trajectory, dtype=object))
        save_scene("assembled_scene.usda")
        assert report["passed"], "Stability validation did not pass."
        log("ASSEMBLY AND STABILITY VALIDATION PASSED")
except Exception:
    import traceback

    report["error"] = traceback.format_exc()
    try:
        report["failure_poses"] = {name: [*pose(name)[0].tolist(), *quat(pose(name)[1]).tolist()] for name in NAMES}
    except Exception as pose_error:
        report["failure_pose_error"] = str(pose_error)
    (OUT / "failed_report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / "failed_trajectory.npz", frames=np.array(trajectory, dtype=object))
    traceback.print_exc()
    if writer:
        writer.close()
    sys.stdout.flush()
    os._exit(1)
if writer:
    writer.close()
sys.stdout.flush()
app.close()
