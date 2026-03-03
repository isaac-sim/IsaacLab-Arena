# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""
CuRobo pick-and-place CLI arguments only.

Lightweight module (no torch, no isaaclab) so run_droid_v2_tabletop_curobo_pick_place.py
can import it before SimulationAppContext and avoid initializing Warp/GPU too early.
"""

from __future__ import annotations

import argparse


def add_script_args(parser: argparse.ArgumentParser) -> None:
    """Add CuRobo pick-and-place CLI arguments."""
    parser.add_argument(
        "--pick_order",
        nargs="+",
        type=str,
        default=None,
        help="Explicit object pick order. If not specified, auto-discovers from scene.",
    )
    parser.add_argument("--max_objects", type=int, default=None, help="Optional limit on number of objects to pick.")
    parser.add_argument("--grasp_z_offset", type=float, default=0.02, help="Z offset above object for grasp (m).")
    parser.add_argument("--place_z_offset", type=float, default=0.12, help="Z offset above bin for place (m).")
    parser.add_argument(
        "--grasp_orientation",
        type=str,
        default="top_down",
        choices=["top_down", "object_aligned", "object_yaw"],
        help="Grasp orientation strategy.",
    )
    parser.add_argument("--bin_half_x", type=float, default=0.08, help="Usable bin half-width X (m).")
    parser.add_argument("--bin_half_y", type=float, default=0.10, help="Usable bin half-width Y (m).")
    parser.add_argument(
        "--gripper_settle_steps",
        type=int,
        default=100,
        help="Sim steps to hold pose while opening/closing gripper.",
    )
    parser.add_argument(
        "--post_place_clearance",
        type=float,
        default=0.10,
        help="Z clearance above place pose after release (0 to skip).",
    )
    parser.add_argument("--approach_distance", type=float, default=0.04, help="CuRobo approach distance (m).")
    parser.add_argument("--retreat_distance", type=float, default=0.06, help="CuRobo retreat distance (m).")
    parser.add_argument("--time_dilation_factor", type=float, default=0.6, help="CuRobo time dilation factor.")
    parser.add_argument("--debug_planner", action="store_true", default=False, help="Enable planner debugging.")
    parser.add_argument("--dump_spheres_dir", type=str, default=None, help="Directory to dump CuRobo collision spheres.")
    parser.add_argument("--dump_spheres_png", action="store_true", default=False, help="Save sphere PNGs.")
    parser.add_argument(
        "--debug_goal",
        action="store_true",
        default=False,
        help="Print goal vs achieved EEF pose after each plan execution.",
    )
    parser.add_argument("--run_sanity_check", action="store_true", default=False, help="Run pre-flight lift check.")
    parser.add_argument("--rerun_recording_path", type=str, default=None, help="Optional .rrd output path for Rerun.")


def add_script_args_to_subparsers(parser: argparse.ArgumentParser) -> None:
    """Register script args on all environment subparsers."""
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                add_script_args(subparser)
