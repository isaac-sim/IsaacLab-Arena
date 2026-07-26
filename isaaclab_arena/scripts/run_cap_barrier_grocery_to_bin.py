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
from isaaclab_arena.integrations.cap_barrier.grocery_physical_result import (
    make_grocery_physical_result_observer,
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
            "publishes one post-reset exterior_cam RGB-D snapshot while it serves "
            "the control barrier."
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
    parser.add_argument(
        "--diagnostic-can-away",
        action="store_true",
        help=(
            "Diagnostic counterfactual: keep the scene unchanged except for moving "
            "the can outside the grasp path. Never use this mode for acceptance."
        ),
    )
    parser.add_argument(
        "--result-request-file",
        default=None,
        metavar="PATH",
        help=(
            "Absolute one-shot request path. When this regular file appears, "
            "evaluate the live grocery physical outcome on the Kit serve thread."
        ),
    )
    parser.add_argument(
        "--result-jsonl",
        default=None,
        metavar="PATH",
        help=(
            "Absolute output path for the atomic one-row physical-result JSONL. "
            "Must be supplied together with --result-request-file."
        ),
    )


def _environment_factory(
    camera_profile: str,
    *,
    diagnostic_can_away: bool = False,
):
    from isaaclab_arena.integrations.cap_barrier.franka_env import make_cap_grocery_to_bin_environment

    keywords = {"camera_profile": camera_profile}
    if diagnostic_can_away:
        keywords["diagnostic_can_away"] = True
    return partial(make_cap_grocery_to_bin_environment, **keywords)


def _scene_ready_marker(
    camera_profile: str,
    *,
    diagnostic_can_away: bool = False,
) -> str:
    can_state = "away-diagnostic" if diagnostic_can_away else "present"
    return (
        "CAP_GROCERY_TO_BIN_SCENE_READY "
        f"object={CAP_GROCERY_OBJECT_ASSET} "
        f"bin={CAP_GROCERY_BIN_ASSET} "
        f"camera={CAP_GROCERY_CAMERA_NAME} "
        f"camera_profile={camera_profile} "
        f"can_state={can_state}"
    )


def _physical_result_observer_factory(args_cli):
    request_path = args_cli.result_request_file
    result_path = args_cli.result_jsonl
    if (request_path is None) != (result_path is None):
        raise ValueError(
            "--result-request-file and --result-jsonl must be supplied together"
        )
    if request_path is None:
        return None
    if not os.path.isabs(request_path) or not os.path.isabs(result_path):
        raise ValueError(
            "--result-request-file and --result-jsonl must be absolute paths"
        )
    return partial(
        make_grocery_physical_result_observer,
        request_path=request_path,
        result_path=result_path,
    )


def _run_grocery(args_cli, *, context_factory=SimulationAppContext, serve=_run_serve) -> None:
    physics_observer_factory = _physical_result_observer_factory(args_cli)
    # Both supported profiles spawn exterior_cam through AppLauncher.
    args_cli.enable_cameras = True
    with context_factory(args_cli):
        serve(
            args_cli.device,
            perception_stream=args_cli.perception_stream,
            # The graph consumes one static observation after ResetEpisode.
            # Keep RTX outside the 200 Hz control loop once that generation's
            # snapshot has been captured.
            perception_frames_per_generation=1,
            # The connector establishes authority by resetting generation 1.
            # Do not publish a fresh pre-reset frame that the generation-less
            # perception transport could briefly expose after ResetEpisode.
            perception_first_capture_generation=2,
            serve_seconds=args_cli.serve_seconds,
            environment_factory=_environment_factory(
                args_cli.camera,
                diagnostic_can_away=args_cli.diagnostic_can_away,
            ),
            initial_gripper_closed=False,
            ready_marker=_scene_ready_marker(
                args_cli.camera,
                diagnostic_can_away=args_cli.diagnostic_can_away,
            ),
            physics_observer_factory=physics_observer_factory,
        )


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    _add_grocery_arguments(parser)
    args_cli = parser.parse_args()
    if not math.isfinite(args_cli.serve_seconds) or args_cli.serve_seconds <= 0.0:
        parser.error("--serve-seconds must be finite and positive")
    try:
        _physical_result_observer_factory(args_cli)
    except ValueError as error:
        parser.error(str(error))
    _run_grocery(args_cli)


if __name__ == "__main__":
    main()
