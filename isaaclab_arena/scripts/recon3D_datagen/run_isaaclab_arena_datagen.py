# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ── Hyperparameters (edit these before running) ─────────────────────────────
SCENE_NAME = "dynamic_balls"   # environment; use --enable_cameras for static camera
OBJECT_NAME = "cracker_box"     # object variant (depends on scene)

CAM0 = {
    "position": (0.0, -0.737, 1.0),   # world-frame (x, y, z)
    "target": (0.466, -0.737, 0.4),   # look-at point in world frame
    "width": 640,
    "height": 480,
    "focal_length": 24.0,              # mm
}
# Uncomment / add more cameras for multi-view capture:
CAM1 = {
    "position": (0.0, -0.337, 0.8),
    "target": (0.466, -0.737, 0.4),
    "width": 600,
    "height": 400,
    "focal_length": 12.0,
}
CAMERAS = [CAM0, CAM1]  # e.g. [CAM0, CAM1] for multi-view

OUTPUT_DIR = "/workspaces/isaaclab_arena/isaaclab_arena/scripts/recon3D_datagen/results/tmp"
NUM_STEPS = 30
OCCLUSION_TOL = 0.1             # depth tolerance for visible-now mask (metres)
ANCHOR_FRAMES = [0]             # frame indices for anchored 3D flow (e.g. [0, 4, 6])

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
    IsaacLabArenaCameraHandler,
    create_static_camera,
)
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import (
    IsaacLabArenaWriter,
    camera_id_from_index,
)

camera_handlers = []
camera_ids = []
for cam_idx, cam_cfg in enumerate(CAMERAS):
    handler = create_static_camera(
        position=cam_cfg["position"],
        target=cam_cfg["target"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        focal_length=cam_cfg["focal_length"],
        prim_path=f"/World/StaticCamera_{cam_idx}",
    )
    camera_handlers.append(handler)
    camera_ids.append(camera_id_from_index(cam_idx))

writer = IsaacLabArenaWriter(OUTPUT_DIR)

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
        obs, _, terminated, truncated, _ = env.step(actions)

        for handler, cam_id in zip(camera_handlers, camera_ids):
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

            # ── Adjacent-frame flow (N-1 files, indices 0 .. N-2) ─
            if step_idx > 0:
                flow_result = handler.compute_exact_scene_flow(env)
                prev_idx = step_idx - 1
                writer.write_optical_flow(
                    handler.get_optical_flow(), cam_id, prev_idx,
                    camera_name=cam_name)
                if flow_result is not None:
                    writer.write_scene_flow_3d(
                        flow_result.scene_flow_3d, cam_id, prev_idx,
                        camera_name=cam_name)
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
                writer.write_flow3d_from_first(
                    ff.flow3d_from_first, cam_id, step_idx,
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

# %%
# Visualization of the generated data
import matplotlib.pyplot as plt

from isaaclab_arena.scripts.recon3D_datagen.datagen_visualizer import (
    visualize_all_modalities_grid,
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
        num_samples=NUM_VIZ_SAMPLES, depth_cmap="Spectral",
        save_path=os.path.join(viz_dir, "data_vis.png"),
    )
    plt.show()
    plt.close("all")

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
