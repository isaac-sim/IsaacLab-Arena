# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the calibrated grocery-to-bin scene for an external GaP ROS policy.

This is a scene-specialized wrapper around the proven move-to-pose barrier
producer. It keeps the arena_droid_b1 base at the identity calibration, spawns
the exact alphabet-soup-can and grey-bin layout from a successful DROID
pick/place, streams RGB-D perception, and otherwise preserves the 200 Hz
generation-fenced serve/reset-follow protocol.
"""

from __future__ import annotations

import math
import os
from functools import partial

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_CAMERA_NAME,
    CAP_GROCERY_CAMERA_PROFILES,
    CAP_GROCERY_OBJECT_ASSET,
)
from isaaclab_arena.scripts.run_cap_barrier_move_to_pose_serve import _SERVE_TIMEOUT_S, _run_serve
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _add_grocery_arguments(parser) -> None:
    parser.add_argument(
        "--perception-stream",
        required=True,
        metavar="HOST:PORT",
        help=(
            "Required cap_perception_bridge endpoint. The grocery producer always "
            "streams exterior_cam RGB-D while it serves the control barrier."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=CAP_GROCERY_CAMERA_PROFILES,
        default="libero",
        help=(
            "Exterior camera profile. 'libero' is the required top-down default; "
            "'oblique' is the pre-authorized open-vocabulary fallback."
        ),
    )
    parser.add_argument(
        "--serve-seconds",
        type=float,
        default=float(os.environ.get("CAP_SERVE_SECONDS", _SERVE_TIMEOUT_S)),
        help=(
            "Bounded serve window in seconds (default 600, env CAP_SERVE_SECONDS). "
            "Raise it when a cold external policy load would otherwise consume "
            "most of the window."
        ),
    )


def _environment_factory(camera_profile: str):
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_grocery_to_bin_environment

    return partial(
        make_cap_grocery_to_bin_environment,
        camera_profile=camera_profile,
    )


def _scene_ready_marker(camera_profile: str) -> str:
    return (
        "CAP_GROCERY_TO_BIN_SCENE_READY "
        f"object={CAP_GROCERY_OBJECT_ASSET} "
        f"bin={CAP_GROCERY_BIN_ASSET} "
        f"camera={CAP_GROCERY_CAMERA_NAME} "
        f"camera_profile={camera_profile}"
    )


def _run_grocery(args_cli, *, context_factory=SimulationAppContext, serve=_run_serve) -> None:
    # Both supported profiles spawn exterior_cam through AppLauncher.
    args_cli.enable_cameras = True
    with context_factory(args_cli):
        serve(
            args_cli.device,
            perception_stream=args_cli.perception_stream,
            serve_seconds=args_cli.serve_seconds,
            environment_factory=_environment_factory(args_cli.camera),
            initial_gripper_closed=False,
            ready_marker=_scene_ready_marker(args_cli.camera),
        )


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    _add_grocery_arguments(parser)
    args_cli = parser.parse_args()
    if not math.isfinite(args_cli.serve_seconds) or args_cli.serve_seconds <= 0.0:
        parser.error("--serve-seconds must be finite and positive")
    _run_grocery(args_cli)


if __name__ == "__main__":
    main()
