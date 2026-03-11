# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ── Hyperparameters (edit these before running) ─────────────────────────────
import math

SCENE_NAME = "dynamic_balls"   # environment; use --enable_cameras for static camera
OBJECT_NAME = "cracker_box"     # object variant (depends on scene)

NUM_STEPS = 30

# Camera coordinates: use a single (x, y, z) tuple for a static coordinate,
# or a list of NUM_STEPS tuples for a dynamic trajectory.  Position and
# target are independent — each can be static or dynamic per camera.

CAM0 = {
    "position": (0.0, -0.737, 1.0),   # static world-frame position
    "target": (0.466, -0.737, 0.4),   # static look-at point
    "width": 640,
    "height": 480,
    "focal_length": 24.0,             # mm
}

# CAM1: dynamic position — horizontal slide to the right, static target
_c1_x_start = -2.0
_c1_x_end = 0.4
CAM1 = {
    "position": [
        (
            _c1_x_start + (_c1_x_end - _c1_x_start) * t / max(NUM_STEPS - 1, 1),
            -0.337,
            0.8,
        )
        for t in range(NUM_STEPS)
    ],
    "target": (0.466, -0.737, 0.4),   # static look-at point
    "width": 600,
    "height": 400,
    "focal_length": 12.0,
}

CAMERAS = [CAM0, CAM1] # CAM1, CAM2... for multi-view

OUTPUT_DIR = "/workspaces/isaaclab_arena/isaaclab_arena/scripts/recon3D_datagen/results/tmp"
OCCLUSION_TOL = 0.1             # depth tolerance for visible-now mask (metres)
ANCHOR_FRAMES = [0]             # frame indices for anchored 3D flow (e.g. [0, 4, 6])
DYNAMIC_MOTION_EPS = 1e-4       # motion threshold for dynamic object detection (m / rad)

# Visualization (optional step at the end)
NUM_VIZ_SAMPLES = 8
SCENE_FLOW_VIZ_FRAME = 0
# ───────────────────────────────────────────────────────────────────────────

# %%
import os
import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher(enable_cameras=True)


# %%
from isaaclab_arena.utils.reload_modules import reload_arena_modules

reload_arena_modules()
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

args_parser = get_isaaclab_arena_environments_cli_parser()

args_cli = args_parser.parse_args([
    "--enable_cameras",
    SCENE_NAME,
    "--object",
    OBJECT_NAME,
])

arena_builder = get_arena_builder_from_cli(args_cli)
env = arena_builder.make_registered()
env.reset()

# %%
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_camera_handler import (
    DynamicObjectTracker,
    IsaacLabArenaCameraHandler,
    ObjectInstanceRegistry,
    create_static_camera,
)
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import (
    IsaacLabArenaWriter,
    camera_id_from_index,
)

def _resolve_coord(coord, step_idx=0):
    """Return the (x, y, z) tuple for *step_idx* (pass-through for static)."""
    return coord if isinstance(coord, tuple) else coord[step_idx]

for _ci, _cc in enumerate(CAMERAS):
    for _key in ("position", "target"):
        _val = _cc[_key]
        if not isinstance(_val, tuple):
            assert len(_val) == NUM_STEPS, (
                f"Camera {_ci} '{_key}' has {len(_val)} entries but "
                f"NUM_STEPS={NUM_STEPS}. Dynamic coordinates must match."
            )

shared_registry = ObjectInstanceRegistry()

camera_handlers = []
camera_ids = []
for cam_idx, cam_cfg in enumerate(CAMERAS):
    handler = create_static_camera(
        position=_resolve_coord(cam_cfg["position"], 0),
        target=_resolve_coord(cam_cfg["target"], 0),
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        focal_length=cam_cfg["focal_length"],
        prim_path=f"/World/StaticCamera_{cam_idx}",
        instance_registry=shared_registry,
    )
    camera_handlers.append(handler)
    camera_ids.append(camera_id_from_index(cam_idx))

writer = IsaacLabArenaWriter(OUTPUT_DIR)
dynamic_tracker = DynamicObjectTracker(shared_registry, num_steps=NUM_STEPS)

# %%
dt = env.unwrapped.step_dt

# Warm-up: one extra sim step so render buffers are initialised.
# The first render after env.reset() may contain stale data.
with torch.inference_mode():
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env.step(actions)
    for handler in camera_handlers:
        handler.update(dt)

sorted_anchor_frames = sorted(ANCHOR_FRAMES)
anchor_frames_set = set(ANCHOR_FRAMES)

for step_idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

        for handler, cam_id, cam_cfg in zip(camera_handlers, camera_ids, CAMERAS):
            pos = cam_cfg["position"]
            tgt = cam_cfg["target"]
            if not isinstance(pos, tuple) or not isinstance(tgt, tuple):
                handler.set_world_pose(
                    _resolve_coord(pos, step_idx),
                    _resolve_coord(tgt, step_idx),
                )
            handler.update(dt)
            cam_name = handler.camera_name

            # ── Static modalities (N files, indices 0 .. N-1) ─────
            writer.write_rgb(
                handler.get_rgb(), cam_id, step_idx, camera_name=cam_name)
            writer.write_depth(
                handler.get_depth(), cam_id, step_idx, camera_name=cam_name)
            writer.write_intrinsics(
                handler.get_intrinsics(), cam_id, step_idx, camera_name=cam_name)
            writer.write_extrinsics(
                handler.get_extrinsics(), cam_id, step_idx, camera_name=cam_name)
            writer.write_normals(
                handler.get_normals(), cam_id, step_idx, camera_name=cam_name)
            seg_data, semantic_info = handler.get_object_instance_segmentation(env)
            writer.write_semantic_segmentation(
                seg_data, semantic_info, cam_id, step_idx,
                camera_name=cam_name,
            )
            dynamic_tracker.register_visible_objects(semantic_info)

            # ── Adjacent-frame flow (N-1 files, indices 0 .. N-2) ─
            if step_idx > 0:
                flow_result = handler.compute_exact_scene_flow(env)
                prev_idx = step_idx - 1
                sf3d = flow_result.scene_flow_3d if flow_result is not None else None
                true_flow = handler.compute_true_optical_flow(sf3d)
                if true_flow is not None:
                    writer.write_optical_flow(
                        true_flow, cam_id, prev_idx,
                        camera_name=cam_name)
                if flow_result is not None:
                    cam_sf = handler.world_to_camera_scene_flow(
                        flow_result.scene_flow_3d)
                    writer.write_scene_flow_3d(
                        cam_sf if cam_sf is not None else flow_result.scene_flow_3d,
                        cam_id, prev_idx, camera_name=cam_name)
                    if flow_result.scene_flow_track_type is not None:
                        writer.write_scene_flow_track_type(
                            flow_result.scene_flow_track_type, cam_id, prev_idx,
                            camera_name=cam_name)

            handler.cache_scene_flow_frame(env)

            # ── Anchor-frame-anchored trajectory flow ─────────────
            if step_idx in anchor_frames_set:
                handler.init_anchor_frame(env, anchor_frame=step_idx)

            for af in sorted_anchor_frames:
                if af > step_idx:
                    break
                ff = handler.compute_anchor_frame_flow(
                    env, anchor_frame=af, occlusion_tol=OCCLUSION_TOL)
                cam_ff = handler.world_to_camera_anchor_flow(
                    ff.points_world_k, anchor_frame=af)
                writer.write_flow3d_from_first(
                    cam_ff if cam_ff is not None else ff.flow3d_from_first,
                    cam_id, step_idx,
                    camera_name=cam_name, anchor_frame=af)
                writer.write_trackable_mask(
                    ff.trackable_mask, cam_id, step_idx,
                    camera_name=cam_name, anchor_frame=af)
                writer.write_in_frame_mask(
                    ff.in_frame_mask, cam_id, step_idx,
                    camera_name=cam_name, anchor_frame=af)
                writer.write_visible_now_mask(
                    ff.visible_now_mask, cam_id, step_idx,
                    camera_name=cam_name, anchor_frame=af)

        dynamic_tracker.record_step_poses(env, step_idx)

writer.write_dynamic_object_poses(
    dynamic_tracker.get_dynamic_object_data(motion_eps=DYNAMIC_MOTION_EPS)
)

# %%
# Visualization of the generated data
import matplotlib.pyplot as plt

from isaaclab_arena.scripts.recon3D_datagen.datagen_visualizer import (
    visualize_all_modalities_grid,
    visualize_all_modalities_video,
    visualize_camera_trajectory,
    visualize_first_frame_flow_3d,
    visualize_scene_flow_3d,
)

for cam_id in camera_ids:
    viz_dir = os.path.join(OUTPUT_DIR, cam_id, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"── {cam_id} ──")

    visualize_all_modalities_grid(
        OUTPUT_DIR, cam_id,
        num_samples=NUM_VIZ_SAMPLES, depth_cmap="Spectral_r",
        save_path=os.path.join(viz_dir, "data_vis.png"),
    )
    plt.show()
    plt.close("all")

    visualize_all_modalities_video(
        OUTPUT_DIR, cam_id,
        fps=5, depth_cmap="Spectral_r",
        save_path=os.path.join(viz_dir, "data_vis.mp4"),
    )

    visualize_camera_trajectory(
        OUTPUT_DIR, cam_id,
        axis_length=0.05, frustum_scale=0.04, num_frustums=NUM_VIZ_SAMPLES,
        save_path=os.path.join(viz_dir, "camera_trajectory_3d.png"),
    )
    plt.show()
    plt.close("all")

    visualize_scene_flow_3d(
        OUTPUT_DIR, cam_id,
        frame_index=SCENE_FLOW_VIZ_FRAME, stride=8, arrow_scale=1.0,
        save_path=os.path.join(viz_dir, f"scene_flow_3d_frame{SCENE_FLOW_VIZ_FRAME}.html"),
    )

    last_frame = NUM_STEPS - 1
    for af in sorted(ANCHOR_FRAMES):
        if af >= NUM_STEPS:
            continue
        target = max(last_frame, af)
        visualize_first_frame_flow_3d(
            OUTPUT_DIR, cam_id,
            frame_index=target, anchor_frame=af, stride=8, arrow_scale=1.0,
            save_path=os.path.join(viz_dir, f"anchor{af}_flow_3d_frame{target}.html"),
        )

    print(f"Visualizations saved to {viz_dir}")

# %%
