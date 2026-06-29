# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
r"""Standalone Isaac Lab Arena data-generation pipeline.

Steps a datagen scene with zero actions and records per-frame camera data
(RGB, depth, normals, semantics, optical/scene flow) plus dynamic-object poses
and mesh samples into a single ``dataset.h5``.

Usage::

    python -m isaaclab_arena_datagen.run_datagen \
        --enable_cameras --headless \
        --output-dir /datasets/dynamic_scenes/miscellaneous/ball_box_robot \
        ball_box_robot

See :mod:`isaaclab_arena_datagen.generate_all_scenes` for the loop that writes
every scene into the canonical ``/datasets/dynamic_scenes`` layout, and
``isaaclab_arena_datagen/README.md`` for the full reference.

To instead collect this same data *while a policy runs*, use
``isaaclab_arena.evaluation.policy_runner`` with ``--collect-datagen`` (see the
README); that path reuses :func:`isaaclab_arena_datagen.pipeline.record_camera_step`.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# Imported before AppLauncher so the parser can use these as defaults.
from isaaclab_arena_datagen.utils.constants import DEFAULT_ROTATION_EPS_RAD, DEFAULT_TRANSLATION_EPS_M


def build_datagen_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the standalone data-generation pipeline."""
    parser = argparse.ArgumentParser(description="Isaac Lab Arena data generation pipeline.")

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("scene_name", type=str, help="Datagen scene name (e.g. 'ball_box_robot')")

    gen_group = parser.add_argument_group("Data Generation")
    gen_group.add_argument("--output-dir", type=str, required=True, help="Output directory for generated data")
    gen_group.add_argument("--num-steps", type=int, default=30, help="Number of simulation steps (default: 30)")
    gen_group.add_argument(
        "--dynamic-translation-eps",
        type=float,
        default=DEFAULT_TRANSLATION_EPS_M,
        help=(
            "Per-step translation threshold (metres) for dynamic-object detection "
            f"(default: {DEFAULT_TRANSLATION_EPS_M})."
        ),
    )
    gen_group.add_argument(
        "--dynamic-rotation-eps",
        type=float,
        default=DEFAULT_ROTATION_EPS_RAD,
        help=(
            f"Per-step rotation threshold (radians) for dynamic-object detection (default: {DEFAULT_ROTATION_EPS_RAD})."
        ),
    )
    gen_group.add_argument(
        "--mesh-sample-spacing",
        type=float,
        default=0.01,
        help="Mesh surface sample spacing in metres (default: 0.01)",
    )

    camera_group = parser.add_argument_group("Camera Resolution")
    camera_group.add_argument(
        "--width", type=int, default=640, help="Image width in pixels, shared by all cameras (default: 640)"
    )
    camera_group.add_argument(
        "--height", type=int, default=480, help="Image height in pixels, shared by all cameras (default: 480)"
    )

    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--num-visualization-samples", type=int, default=8, help="Visualization grid samples (default: 8)"
    )
    viz_group.add_argument(
        "--scene-flow-visualization-frame",
        type=int,
        default=0,
        help="Frame index for scene flow 3D visualization (default: 0)",
    )
    viz_group.add_argument(
        "--visualizations",
        action="store_true",
        help="Run DatagenVisualizer to produce per-scene PNG grid, MP4, and HTML plots.",
    )

    # AppLauncher args (provides --headless, --enable_cameras, --device, ...).
    AppLauncher.add_app_launcher_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Parse CLI args and launch AppLauncher before importing Omniverse-dependent
# modules.  Cameras are force-enabled because the pipeline always renders
# sensor data.  ``_ARGS`` is reused in ``main`` to avoid double-parsing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _ARGS = build_datagen_parser().parse_args()
    _ARGS.enable_cameras = True
    print("Launching simulation app")
    _APP_LAUNCHER = AppLauncher(_ARGS)

# Post-SimulationApp imports -- these require AppLauncher / SimulationApp running.
from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app  # noqa: E402
from isaaclab_arena_datagen.pipeline import (  # noqa: E402
    SimDataCollectionSetup,
    run_simulation_loop,
    save_dynamic_objects,
)


def main() -> None:
    """Run the full data generation pipeline: setup, simulate, save, visualize."""
    args = _ARGS

    setup = SimDataCollectionSetup.from_config(
        scene_name=args.scene_name,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        width=args.width,
        height=args.height,
    )

    run_simulation_loop(
        setup.env,
        setup.camera_setups,
        setup.writer,
        setup.dynamic_tracker,
        args.num_steps,
    )

    save_dynamic_objects(
        setup.env,
        setup.writer,
        setup.dynamic_tracker,
        args.dynamic_translation_eps,
        args.dynamic_rotation_eps,
        args.mesh_sample_spacing,
    )

    setup.writer.close()

    if args.visualizations:
        from isaaclab_arena_datagen.visualizer import DatagenVisualizer

        hdf5_path = os.path.join(args.output_dir, "dataset.h5")
        DatagenVisualizer(
            args.output_dir,
            [cam.camera_id for cam in setup.camera_setups],
            num_visualization_samples=args.num_visualization_samples,
            scene_flow_visualization_frame=args.scene_flow_visualization_frame,
            num_steps=args.num_steps,
            hdf5_path=hdf5_path,
        ).generate_all()

    teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)


if __name__ == "__main__":
    main()
