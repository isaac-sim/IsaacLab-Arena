# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record an mp4 video of the rollout (uses gymnasium.wrappers.RecordVideo).",
    )
    parser.add_argument(
        "--video_dir",
        "--video-dir",
        type=str,
        default="/eval/videos",
        help="Output directory for recorded videos. Created if missing. Used with --video and/or --camera_video.",
    )
    parser.add_argument(
        "--camera_video",
        "--camera-video",
        action="store_true",
        default=False,
        help=(
            "Record one mp4 per camera in obs['camera_obs'] (what the policy actually sees)."
            " Independent of --video; use either or both."
        ),
    )
    _add_datagen_collection_arguments(parser)


def _add_datagen_collection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add opt-in datagen data-collection arguments.

    When ``--collect-datagen`` is set, the rollout records SyntheticScene-format
    data (RGB/depth/normals/semantics/flow + dynamic-object poses) from dedicated
    cameras into ``{--datagen-output-dir}/dataset.h5`` via
    ``isaaclab_arena_datagen.collection.collector.DatagenCollector``. Requires
    ``--enable_cameras`` and a fixed horizon (``--num_steps``). Off by default;
    rollout behavior is unchanged unless the flag is passed.
    """
    group = parser.add_argument_group("Datagen collection")
    group.add_argument(
        "--collect-datagen",
        "--collect_datagen",
        action="store_true",
        default=False,
        help="Collect SyntheticScene-format datagen data during the rollout (requires --enable_cameras).",
    )
    group.add_argument(
        "--datagen-output-dir",
        "--datagen_output_dir",
        type=str,
        default="/eval/datagen",
        help="Output directory for the datagen dataset.h5 (used with --collect-datagen).",
    )
    group.add_argument(
        "--datagen-width", "--datagen_width", type=int, default=640, help="Datagen camera image width (px)."
    )
    group.add_argument(
        "--datagen-height", "--datagen_height", type=int, default=480, help="Datagen camera image height (px)."
    )
    group.add_argument(
        "--datagen-mesh-sample-spacing",
        "--datagen_mesh_sample_spacing",
        type=float,
        default=0.01,
        help="Datagen mesh surface sample spacing in metres.",
    )
    group.add_argument(
        "--datagen-camera-position",
        "--datagen_camera_position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "World-frame position of the datagen camera. If set (with --datagen-camera-target),"
            " overrides the environment's get_default_cameras / the default fallback view."
        ),
    )
    group.add_argument(
        "--datagen-camera-target",
        "--datagen_camera_target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame look-at point of the datagen camera. Used with --datagen-camera-position.",
    )
    group.add_argument(
        "--datagen-focal-length",
        "--datagen_focal_length",
        type=float,
        default=24.0,
        help="Datagen camera focal length in mm (used with --datagen-camera-position/-target).",
    )
