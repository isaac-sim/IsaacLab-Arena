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
from isaaclab_arena.scripts.recon3D_datagen.isaaclab_arena_writer import IsaacLabArenaWriter

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

for step_idx in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)

        # Refresh the static camera sensor after the simulation step
        camera_handler.update(dt)

        # Skip the first frame (render data may be uninitialised)
        if step_idx == 0:
            continue

        # Write RGB, depth, intrinsics, and extrinsics to disk
        camera_name = camera_handler.camera_name
        writer.write_frame(
            rgb=camera_handler.get_rgb(),
            depth=camera_handler.get_depth(),
            intrinsics=camera_handler.get_intrinsics(),
            extrinsics=camera_handler.get_extrinsics(),
            camera_name=camera_name,
            frame_index=step_idx,
        )

# %%
# Visualization of the generated data
from isaaclab_arena.scripts.recon3D_datagen.datagen_visualizer import (
    visualize_camera_trajectory,
    visualize_depth_grid,
    visualize_rgb_grid,
)

camera_name = camera_handler.camera_name
viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(viz_dir, exist_ok=True)
num_samples = 8

visualize_rgb_grid(
    OUTPUT_DIR,
    camera_name,
    num_samples=num_samples,
    save_path=os.path.join(viz_dir, "rgb_grid.png"),
)

visualize_depth_grid(
    OUTPUT_DIR,
    camera_name,
    num_samples=num_samples,
    cmap="Spectral",
    save_path=os.path.join(viz_dir, "depth_grid.png"),
)

visualize_camera_trajectory(
    OUTPUT_DIR,
    camera_name,
    axis_length=0.05,
    frustum_scale=0.04,
    num_frustums=num_samples,
    save_path=os.path.join(viz_dir, "camera_trajectory_3d.png"),
)

print(f"Visualizations of {num_samples} frames saved to {viz_dir}")

# %%
