# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Capture placement GIFs for homogeneous and heterogeneous object placement.

Extends the ``pick_and_place_maple_table`` environment setup (same lighting,
HDR, table reference) to generate animated GIFs comparing homogeneous vs
heterogeneous placement across parallel environments.

Uses the Isaac Lab ``Camera`` sensor directly (same pattern as the official
``run_usd_camera.py`` tutorial) for reliable headless capture, then tiles
per-env images into a grid.

Usage (inside Docker)::

    # Homogeneous
    /isaac-sim/python.sh isaaclab_arena_examples/relations/placement_gif_capture.py \\
        --mode homogeneous --num_envs 4 --num_resets 10 --enable_cameras --headless \\
        --output_dir /workspaces/isaaclab_arena/docs/images/placement_gifs

    # Heterogeneous
    /isaac-sim/python.sh isaaclab_arena_examples/relations/placement_gif_capture.py \\
        --mode heterogeneous --num_envs 4 --num_resets 10 --enable_cameras --headless \\
        --output_dir /workspaces/isaaclab_arena/docs/images/placement_gifs
"""

from isaaclab.app import AppLauncher

AppLauncher()

# %%


HOMOGENEOUS_OBJECTS = ["cracker_box", "mug", "tomato_soup_can", "sugar_box"]

HETEROGENEOUS_VARIANTS = ["cracker_box", "mug", "tomato_soup_can", "sugar_box"]
COMMON_EXTRA_OBJECTS = ["dex_cube", "red_container"]

LIGHT_INTENSITY = 500.0
HDR_NAME = "home_office_robolab"

# Offsets from each env origin.  The table prim sits at ~(0.2, 0.0) relative
# to the env origin; objects cluster around (0.7, -0.1, 0.03).
OBJ_CENTER = (0.7, 0.05, 0.03)
CAM_EYE = (OBJ_CENTER[0] + 0.8, OBJ_CENTER[1], OBJ_CENTER[2] + 1.0)
CAM_LOOKAT = OBJ_CENTER


def _build_base_scene(asset_registry, hdr_registry):
    """Build the base scene matching pick_and_place_maple_table: background, lighting, table anchor."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.relations.relations import IsAnchor

    background = asset_registry.get_asset_by_name("maple_table_robolab")()

    light = asset_registry.get_asset_by_name("light")(
        spawner_cfg=sim_utils.DomeLightCfg(intensity=LIGHT_INTENSITY),
    )
    hdr = hdr_registry.get_hdr_by_name(HDR_NAME)()
    light.add_hdr(hdr)

    table_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
        parent_asset=background,
    )
    table_reference.add_relation(IsAnchor())

    return background, light, table_reference


def _build_homogeneous_scene(asset_registry, hdr_registry):
    """Build a scene with the same objects in every environment."""
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene

    background, light, table_reference = _build_base_scene(asset_registry, hdr_registry)

    objects = []
    for name in HOMOGENEOUS_OBJECTS:
        obj = asset_registry.get_asset_by_name(name)()
        obj.add_relation(On(table_reference))
        objects.append(obj)

    scene = Scene(assets=[background, light, table_reference, *objects])
    return scene


def _build_heterogeneous_scene(asset_registry, hdr_registry):
    """Build a scene where each environment gets a different object variant."""
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene

    background, light, table_reference = _build_base_scene(asset_registry, hdr_registry)

    variant_objects = [asset_registry.get_asset_by_name(n)() for n in HETEROGENEOUS_VARIANTS]
    hetero_object = RigidObjectSet(name="hetero_pick", objects=variant_objects)
    hetero_object.add_relation(On(table_reference))

    extras = []
    for name in COMMON_EXTRA_OBJECTS:
        obj = asset_registry.get_asset_by_name(name)()
        obj.add_relation(On(table_reference))
        extras.append(obj)

    scene = Scene(assets=[background, light, table_reference, hetero_object, *extras])
    return scene


def _create_camera_sensor(num_envs: int, env_spacing: float, cam_width: int, cam_height: int):
    """Create a Camera sensor following the official Isaac Lab run_usd_camera.py pattern.

    Creates one camera per environment, each positioned at the docs camera angle
    relative to its environment origin.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    for i in range(num_envs):
        sim_utils.create_prim(f"/World/CameraMount_{i:02d}", "Xform")

    camera_cfg = CameraCfg(
        prim_path="/World/CameraMount_.*/CameraSensor",
        update_period=0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(cfg=camera_cfg)

    return camera


def _position_cameras(camera, env_origins, device: str):
    """Position cameras at the table-viewing angle, offset per actual environment origin."""
    import torch

    num_envs = env_origins.shape[0]
    positions = []
    targets = []
    for i in range(num_envs):
        ox = env_origins[i, 0].item()
        oy = env_origins[i, 1].item()

        positions.append([ox + CAM_EYE[0], oy + CAM_EYE[1], CAM_EYE[2]])
        targets.append([ox + CAM_LOOKAT[0], oy + CAM_LOOKAT[1], CAM_LOOKAT[2]])

    print(f"  Camera positions: {positions}")
    print(f"  Camera targets:   {targets}")
    camera.set_world_poses_from_view(
        torch.tensor(positions, device=device),
        torch.tensor(targets, device=device),
    )


def _tile_images(images, num_cols=2):
    """Tile a list of images into a grid."""
    import math
    import numpy as np

    n = len(images)
    num_rows = math.ceil(n / num_cols)
    h, w = images[0].shape[:2]

    grid = np.zeros((num_rows * h, num_cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // num_cols
        c = idx % num_cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img[:, :, :3]

    return grid


def run_placement_gif_capture(
    mode: str = "homogeneous",
    num_envs: int = 4,
    output_dir: str = "/tmp/placement_gifs",
    num_resets: int = 10,
    warmup_steps: int = 20,
    gif_frame_duration_ms: int = 1500,
    env_spacing: float = 3.0,
    cam_width: int = 640,
    cam_height: int = 480,
) -> str:
    """Capture placement frames and assemble into an animated GIF.

    Uses the Isaac Lab Camera sensor directly (following the official
    ``run_usd_camera.py`` tutorial pattern) for reliable headless capture.
    """
    import numpy as np
    import os
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry, HDRImageRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    os.makedirs(output_dir, exist_ok=True)
    asset_registry = AssetRegistry()
    hdr_registry = HDRImageRegistry()

    if mode == "homogeneous":
        scene = _build_homogeneous_scene(asset_registry, hdr_registry)
    elif mode == "heterogeneous":
        scene = _build_heterogeneous_scene(asset_registry, hdr_registry)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'homogeneous' or 'heterogeneous'.")

    arena_env = IsaacLabArenaEnvironment(
        name=f"{mode}_placement_gif",
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([
        "--num_envs",
        str(num_envs),
        "--enable_cameras",
        "--env_spacing",
        str(env_spacing),
        "--resolve_on_reset",
    ])

    env_builder = ArenaEnvBuilder(arena_env, args_cli)
    env = env_builder.make_registered(render_mode=None)

    sim = env.unwrapped.sim
    device = env.unwrapped.device

    print("Creating camera sensors...")
    camera = _create_camera_sensor(num_envs, env_spacing, cam_width, cam_height)
    sim.reset()
    camera.update(dt=0.0)

    env_origins = env.unwrapped.scene.env_origins
    _position_cameras(camera, env_origins, device)
    print(f"Camera sensor created: {num_envs} cameras at {cam_width}x{cam_height}")

    # Warm up renderer: run several reset+step cycles so Omniverse fully loads
    # textures, materials, and lighting before we start capturing.
    print("Warming up renderer...")
    for warmup_round in range(5):
        env.reset()
        for _ in range(20):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=device)
                env.step(actions)
            sim.render()
            camera.update(dt=env.unwrapped.step_dt)
    print("Renderer warm-up complete.")

    # Capture extra "burn-in" frames that we discard; the renderer needs a
    # few reset-render cycles after sim.reset() before materials fully resolve.
    burn_in = 3
    total_captures = num_resets + burn_in
    all_frames = []
    for reset_idx in range(total_captures):
        env.reset()

        for _ in range(warmup_steps):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=device)
                env.step(actions)
            sim.render()
            camera.update(dt=env.unwrapped.step_dt)

        rgb = camera.data.output["rgb"]  # (num_envs, H, W, 3 or 4)

        per_env_images = []
        for env_idx in range(num_envs):
            img = rgb[env_idx].cpu().numpy().astype(np.uint8)
            per_env_images.append(img)

        tiled = _tile_images(per_env_images, num_cols=2)
        pixel_sum = int(tiled.sum())
        label = "burn-in" if reset_idx < burn_in else "capture"
        print(f"[{mode}] {label} frame {reset_idx + 1}/{total_captures} (shape={tiled.shape}, sum={pixel_sum})")
        all_frames.append(tiled)

    frames = all_frames[burn_in:]

    env.close()

    if not frames:
        print("No frames captured!")
        return ""

    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    gif_path = os.path.join(output_dir, f"{mode}_placement.gif")
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=gif_frame_duration_ms,
        loop=0,
    )
    print(f"\nSaved GIF: {gif_path}  ({len(frames)} frames, {gif_frame_duration_ms}ms each)")

    for idx, img in enumerate(pil_frames):
        png_path = os.path.join(output_dir, f"{mode}_layout_{idx:02d}.png")
        img.save(png_path)

    return gif_path


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Placement GIF capture (homogeneous / heterogeneous)")
    parser.add_argument("--mode", type=str, default="homogeneous", choices=["homogeneous", "heterogeneous"])
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="/tmp/placement_gifs")
    parser.add_argument("--num_resets", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--gif_frame_duration_ms", type=int, default=1500)
    parser.add_argument("--env_spacing", type=float, default=3.0)
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)
    args, _ = parser.parse_known_args()

    run_placement_gif_capture(
        mode=args.mode,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        num_resets=args.num_resets,
        warmup_steps=args.warmup_steps,
        gif_frame_duration_ms=args.gif_frame_duration_ms,
        env_spacing=args.env_spacing,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
    )

# %%
