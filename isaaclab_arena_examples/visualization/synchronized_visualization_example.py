# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronized visualization demo on the GR1 multi-object no-collision scene.

Builds the gr1_table_multi_object_no_collision environment across parallel envs,
attaches a SynchronizedVisualizer, and writes two videos:

- *_global.mp4: a single camera framing all envs at once.
- *_grid.mp4: one camera per env, tiled into a grid.

The global pose and per-env camera offsets are configurable from the CLI, so this
script doubles as a general-purpose visualization entry point.

Run inside the Docker container:

    /isaac-sim/python.sh \\
        isaaclab_arena_examples/visualization/synchronized_visualization_example.py \\
        --enable_cameras --headless --num_envs 9 --env_spacing 4.0 \\
        --num_steps 120 --reset_every 40 \\
        --output_dir /workspaces/isaaclab_arena/outputs/sync_viz
"""

from __future__ import annotations

import argparse

# pyright: reportAttributeAccessIssue=false


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()

    # -- Environment selection (gr1_table_multi_object_no_collision) ----------
    parser.add_argument("--embodiment", type=str, default="gr1_joint", help="Robot embodiment to use.")
    parser.add_argument(
        "--mode",
        type=str,
        default="homogeneous",
        choices=["homogeneous", "heterogeneous"],
        help="Placement mode for the no-collision scene.",
    )
    parser.add_argument("--objects", nargs="*", type=str, default=None, help="Override object list.")

    # -- Rollout --------------------------------------------------------------
    parser.add_argument("--num_steps", type=int, default=120, help="Number of zero-action steps to run.")
    parser.add_argument("--reset_every", type=int, default=40, help="Reset (re-place objects) every N steps.")
    parser.add_argument("--warmup_steps", type=int, default=15, help="Renderer warm-up steps before capturing.")
    parser.add_argument("--fps", type=int, default=20, help="Output video frames per second.")
    parser.add_argument("--also_gif", action="store_true", default=False, help="Also write animated gifs.")
    parser.add_argument(
        "--max_grid_width",
        type=int,
        default=1920,
        help="Cap the tiled grid width [px]; it is downscaled to fit so large env counts stay reasonable.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspaces/isaaclab_arena/outputs/sync_viz",
        help="Directory for output videos.",
    )

    # -- Per-env view (offsets relative to each env origin) -------------------
    # Defaults frame the GR1 table side-on: camera behind (-Y), elevated and angled down onto the tabletop.
    parser.add_argument("--env_eye_offset", nargs=3, type=float, default=[0.3, -2.8, 2.8], help="Per-env camera eye.")
    parser.add_argument(
        "--env_lookat_offset", nargs=3, type=float, default=[0.3, -0.1, 0.75], help="Per-env camera target."
    )
    parser.add_argument(
        "--env_focal_length",
        type=float,
        default=26.0,
        help="Per-env camera focal length [mm]. Larger zooms in and crops neighboring envs.",
    )

    # -- Global view (world coords; omit eye/lookat to auto-frame) ------------
    parser.add_argument("--global_eye", nargs=3, type=float, default=None, help="Global camera eye (world).")
    parser.add_argument("--global_lookat", nargs=3, type=float, default=None, help="Global camera target (world).")
    parser.add_argument("--global_azimuth", type=float, default=-45.0, help="Auto-frame azimuth [deg].")
    parser.add_argument("--global_elevation", type=float, default=35.0, help="Auto-frame elevation [deg].")
    return parser


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()

    # Launch the simulation app before importing Isaac Lab / Arena modules.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.utils.synchronized_visualizer import EnvView, GlobalView, SynchronizedVisualizer
    from isaaclab_arena_environments.gr1_table_multi_object_no_collision_environment import (
        GR1TableMultiObjectNoCollisionEnvironment,
    )

    # get_env() expects these keys on args_cli; default them when the base CLI omits them.
    args_cli.teleop_device = getattr(args_cli, "teleop_device", None)
    args_cli.episode_length_s = getattr(args_cli, "episode_length_s", None)

    env_view = EnvView(
        eye_offset=tuple(args_cli.env_eye_offset),
        lookat_offset=tuple(args_cli.env_lookat_offset),
        focal_length=args_cli.env_focal_length,
    )
    global_view = GlobalView(
        eye=tuple(args_cli.global_eye) if args_cli.global_eye is not None else None,
        lookat=tuple(args_cli.global_lookat) if args_cli.global_lookat is not None else None,
        azimuth_deg=args_cli.global_azimuth,
        elevation_deg=args_cli.global_elevation,
    )

    arena_env = GR1TableMultiObjectNoCollisionEnvironment().get_env(args_cli)
    builder = ArenaEnvBuilder(arena_env, args_cli)

    # Register the per-env tiled camera on the scene before build so Isaac Lab
    # sets up its tiled render product once and updates it as part of env.step.
    cam_name = SynchronizedVisualizer.ENV_CAM_NAME
    arena_env.scene.add_sensor(cam_name, SynchronizedVisualizer.build_env_camera_cfg(env_view, cam_name))

    env_cfg = builder.compose_manager_cfg()
    # The global view is read via env.render(): its size is the viewer resolution and
    # rgb_array mode is what makes render() return frames. The visualizer routes the
    # render product through its own world camera (the GUI viewport cam is unposable headless).
    env_cfg.viewer.resolution = (global_view.width, global_view.height)
    env = builder.make_registered(env_cfg=env_cfg, render_mode="rgb_array")
    device = env.unwrapped.device

    viz = SynchronizedVisualizer(
        env,
        env_view=env_view,
        global_view=global_view,
        max_grid_width=args_cli.max_grid_width,
    )

    action_shape = env.action_space.shape

    warned_render_none = False

    def _settle(num_steps: int) -> None:
        # RTX annotators return black for the first few frames; discard them so the
        # first captured frame isn't blank. A None render means the global view is
        # misconfigured and every later capture() would skip it, so surface it once
        # (warn-once, since _settle reruns after every reset).
        nonlocal warned_render_none
        for _ in range(num_steps):
            with torch.inference_mode():
                env.step(torch.zeros(action_shape, device=device))
            # env.render() here just surfaces a misconfig (None return) early; the
            # frame itself is discarded during warm-up.
            if env.render() is None and not warned_render_none:
                print(
                    "Warning: env.render() returned None during warm-up; the global view will be empty. "
                    "Build the env with render_mode='rgb_array' and run with --enable_cameras."
                )
                warned_render_none = True

    try:
        print("Warming up renderer...")
        env.reset()
        viz.initialize()
        _settle(args_cli.warmup_steps)

        print(f"Capturing {args_cli.num_steps} steps...")
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                env.step(torch.zeros(action_shape, device=device))
            viz.capture()

            if args_cli.reset_every > 0 and (step + 1) % args_cli.reset_every == 0:
                env.reset()
                viz.reposition()
                _settle(args_cli.warmup_steps)

        written = viz.save(
            args_cli.output_dir,
            fps=args_cli.fps,
            name_prefix="gr1_no_collision",
            also_gif=args_cli.also_gif,
        )
        print("Wrote:")
        for name, path in written.items():
            print(f"  {name}: {path}")
    finally:
        # Always release GPU/sim resources, even if the capture loop raises.
        viz.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
