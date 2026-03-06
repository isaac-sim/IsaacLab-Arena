# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

# Dynamic Balls (--enable_cameras required for the static camera sensor)
args_cli = args_parser.parse_args([
    "--enable_cameras",
    "dynamic_balls",
    "--object",
    "cracker_box",
])

arena_builder = get_arena_builder_from_cli(args_cli)
env = arena_builder.make_registered()
env.reset()

# %%

# ── Static camera & writer configuration ──────────────────────────────
CAMERA_POSITION = (0.0, -0.737, 1.0)   # world-frame position (x, y, z)
CAMERA_TARGET = (0.466, -0.737, 0.4)     # look-at point in world frame
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_LENGTH = 24.0                  # mm
OUTPUT_DIR = "/workspaces/isaaclab_arena/isaaclab_arena/scripts/recon3D_datagen/results/tmp"

from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_camera_handler import (
    IsaacLabArenaCameraHandler,
    create_static_camera,
)
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import (
    IsaacLabArenaWriter,
    camera_id_from_index,
)

camera_handler = create_static_camera(
    position=CAMERA_POSITION,
    target=CAMERA_TARGET,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    focal_length=FOCAL_LENGTH,
)

writer = IsaacLabArenaWriter(OUTPUT_DIR)

# %%

NUM_STEPS = 30
dt = env.unwrapped.step_dt
camera_id = camera_id_from_index(0)
camera_name = camera_handler.camera_name

# Warm-up: one extra sim step so render buffers are initialised.
# The first render after env.reset() may contain stale data.
with torch.inference_mode():
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env.step(actions)
    camera_handler.update(dt)

for step_idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)
        camera_handler.update(dt)

        # ── Static modalities (N files, indices 0 .. N-1) ─────────
        writer.write_rgb(
            camera_handler.get_rgb(), camera_id, step_idx, camera_name=camera_name)
        writer.write_depth(
            camera_handler.get_depth(), camera_id, step_idx, camera_name=camera_name)
        writer.write_intrinsics(
            camera_handler.get_intrinsics(), camera_id, step_idx, camera_name=camera_name)
        writer.write_extrinsics(
            camera_handler.get_extrinsics(), camera_id, step_idx, camera_name=camera_name)
        writer.write_normals(
            camera_handler.get_normals(), camera_id, step_idx, camera_name=camera_name)
        seg_data, _ = camera_handler.get_semantic_segmentation()
        writer.write_semantic_segmentation(
            seg_data, camera_handler.get_semantic_info(),
            camera_id, step_idx, camera_name=camera_name)

        # ── Flow modalities (N-1 files, indices 0 .. N-2) ─────────
        # compute_exact_scene_flow returns displacement from the
        # cached (previous) frame to the current frame.  Rendered
        # motion vectors at step k also describe frame k-1 → k.
        # Both are saved at frame index k-1.
        if step_idx > 0:
            flow_result = camera_handler.compute_exact_scene_flow(env)
            prev_idx = step_idx - 1
            writer.write_optical_flow(
                camera_handler.get_optical_flow(), camera_id, prev_idx,
                camera_name=camera_name)
            if flow_result is not None:
                writer.write_scene_flow_3d(
                    flow_result.scene_flow_3d, camera_id, prev_idx,
                    camera_name=camera_name)
                if flow_result.scene_flow_track_type is not None:
                    writer.write_scene_flow_track_type(
                        flow_result.scene_flow_track_type, camera_id, prev_idx,
                        camera_name=camera_name)

        camera_handler.cache_scene_flow_frame(env)

# %%
# Visualization of the generated data
from isaaclab_arena.scripts.recon3D_datagen.datagen_visualizer import (
    visualize_all_modalities_grid,
    visualize_camera_trajectory,
    visualize_scene_flow_3d,
)

camera_id = camera_id_from_index(0)
# Store visualizations inside each camera folder (e.g. cam0/visualizations/)
viz_dir = os.path.join(OUTPUT_DIR, camera_id, "visualizations")
os.makedirs(viz_dir, exist_ok=True)
num_samples = 8

# Single plot: color, depth, flow2d, normals, semantics per frame
visualize_all_modalities_grid(
    OUTPUT_DIR,
    camera_id,
    num_samples=num_samples,
    depth_cmap="Spectral",
    save_path=os.path.join(viz_dir, "data_vis.png"),
)

# 3D camera trajectory (separate figure)
visualize_camera_trajectory(
    OUTPUT_DIR,
    camera_id,
    axis_length=0.05,
    frustum_scale=0.04,
    num_frustums=num_samples,
    save_path=os.path.join(viz_dir, "camera_trajectory_3d.png"),
)

# Interactive 3D scene flow (saved as rotatable HTML)
frame_index = 0
visualize_scene_flow_3d(
    OUTPUT_DIR,
    camera_id,
    frame_index=frame_index,
    stride=8,
    arrow_scale=1.0,
    save_path=os.path.join(viz_dir, f"scene_flow_3d_frame{frame_index}.html"),
)

print(f"Visualizations saved to {viz_dir}")

# %%
