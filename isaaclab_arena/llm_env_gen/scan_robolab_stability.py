# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scan all robolab objects for stability with and without convexHull override.

Boots the GR1 table environment once per object (homogeneous mode, 1 env),
runs the stability check, and prints a summary table.

Usage (inside container):
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/scan_robolab_stability.py --headless
"""

from __future__ import annotations

import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.llm_env_gen.stability_utils import (
    add_stability_cli_args,
    classify_object,
    get_rigid_pose,
    get_rigid_velocity,
    thresholds_from_args,
    tilt_angle_rad,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app

ROBOLAB_OBJECTS = [
    "alphabet_soup_can_hope_robolab",
    "banana_ycb_robolab",
    "bbq_sauce_bottle_hope_robolab",
    "blue_block_basic_robolab",
    "bowl_ycb_robolab",
    "brick_ycb_robolab",
    "butter_hope_robolab",
    "canned_mushrooms_hope_robolab",
    "canned_peaches_hope_robolab",
    "canned_tuna_hope_robolab",
    "cheez_it_ycb_robolab",
    "chocolate_pudding_mix_hope_robolab",
    "chocolate_pudding_ycb_robolab",
    "clamp_ycb_robolab",
    "coffee_can_ycb_robolab",
    "cordless_drill_ycb_robolab",
    "corn_can_hope_robolab",
    "cream_cheese_hope_robolab",
    "dry_erase_marker_ycb_robolab",
    "granola_bars_hope_robolab",
    "green_beans_can_hope_robolab",
    "green_block_basic_robolab",
    "gregorys_coffee_cup_objaverse_robolab",
    "hammer_handal_robolab",
    "jello_ycb_robolab",
    "ketchup_bottle_hope_robolab",
    "ladle_handal_robolab",
    "lunchbag_objaverse_robolab",
    "macaroni_and_cheese_hope_robolab",
    "mayonnaise_bottle_hope_robolab",
    "measuring_cups_handal_robolab",
    "measuring_spoon_handal_robolab",
    "milk_carton_hope_robolab",
    "mug_ycb_robolab",
    "mustard_bottle_hope_robolab",
    "mustard_ycb_robolab",
    "oatmeal_raisin_cookies_hope_robolab",
    "orange_juice_carton_hope_robolab",
    "parmesan_cheese_canister_hope_robolab",
    "peas_and_carrots_hope_robolab",
    "pineapple_slices_can_hope_robolab",
    "pitcher_ycb_robolab",
    "pitted_cherries_hope_robolab",
    "popcorn_box_hope_robolab",
    "raisin_box_hope_robolab",
    "ranch_dressing_hope_robolab",
    "red_bell_pepper_objaverse_robolab",
    "red_block_basic_robolab",
    "red_onion_fruits_veggies_robolab",
    "salad_tongs_handal_robolab",
    "scissors_ycb_robolab",
    "serving_spoon_handal_robolab",
    "serving_spoons_handal_robolab",
    "snickers_bar_objaverse_robolab",
    "soft_scrub_ycb_robolab",
    "spaghetti_hope_robolab",
    "spam_can_ycb_robolab",
    "spoon_handal_robolab",
    "spoon_1_handal_robolab",
    "spoon_2_handal_robolab",
    "spring_clamp_ycb_robolab",
    "sugar_box_ycb_robolab",
    "tomato_sauce_can_hope_robolab",
    "tomato_soup_can_ycb_robolab",
    "tuna_can_ycb_robolab",
    "wood_block_ycb_robolab",
    "yellow_block_basic_robolab",
    "yogurt_cup_hope_robolab",
]


def run_one_object(args_cli, object_name: str, force_convex_hull: bool) -> dict:
    """Boot the GR1 env with a single object and return stability metrics."""
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    add_stability_cli_args(parser)
    parser = get_isaaclab_arena_environments_cli_parser(parser)

    test_args = parser.parse_args([
        "--headless",
        "--num_envs",
        "1",
        "--settle_steps",
        "60",
        "gr1_table_multi_object_no_collision",
        "--embodiment",
        "gr1_joint",
        "--mode",
        "homogeneous",
        "--objects",
        object_name,
    ])

    # Override force_convex_hull
    arena_builder = get_arena_builder_from_cli(test_args)
    arena_builder.arena_env.force_convex_hull = force_convex_hull

    env, _ = arena_builder.make_registered_and_return_cfg()

    try:
        env.reset()

        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        spawn_pos, spawn_quat = get_rigid_pose(env, object_name, 0)

        env.step(zero)
        t1_pos, _ = get_rigid_pose(env, object_name, 0)

        for _ in range(60):
            env.step(zero)

        now_pos, now_quat = get_rigid_pose(env, object_name, 0)
        lin_vel, ang_vel = get_rigid_velocity(env, object_name, 0)

        thresholds = thresholds_from_args(test_args)
        metrics = {
            "first_step_jump_m": float(torch.linalg.norm(t1_pos - spawn_pos).item()),
            "xy_drift_m": float(torch.linalg.norm((now_pos - spawn_pos)[:2]).item()),
            "z_drop_m": float(max(0.0, (spawn_pos[2] - now_pos[2]).item())),
            "tilt_rad": tilt_angle_rad(spawn_quat, now_quat),
            "lin_vel_norm": float(torch.linalg.norm(lin_vel).item()),
            "ang_vel_norm": float(torch.linalg.norm(ang_vel).item()),
            "aabb_overlap_with": [],
        }
        metrics["status"] = classify_object(metrics, thresholds)
        return metrics
    finally:
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
        env.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    add_stability_cli_args(parser)
    args_cli, _ = parser.parse_known_args()

    results = []

    with SimulationAppContext(args_cli):
        for obj_name in ROBOLAB_OBJECTS:
            row = {"name": obj_name}
            for hull_mode, label in [(False, "decomp"), (True, "hull")]:
                try:
                    metrics = run_one_object(args_cli, obj_name, force_convex_hull=hull_mode)
                    row[f"status_{label}"] = metrics["status"]
                    row[f"tilt_{label}"] = f"{metrics['tilt_rad'] * 57.2958:.1f}"
                    row[f"z_drop_{label}"] = f"{metrics['z_drop_m']:.4f}"
                    row[f"xy_drift_{label}"] = f"{metrics['xy_drift_m']:.4f}"
                except Exception as e:
                    row[f"status_{label}"] = f"ERROR: {e}"
                    row[f"tilt_{label}"] = "N/A"
                    row[f"z_drop_{label}"] = "N/A"
                    row[f"xy_drift_{label}"] = "N/A"

            changed = row.get("status_decomp") != row.get("status_hull")
            marker = " *** CHANGED ***" if changed else ""
            print(
                f"{obj_name:<50s} decomp={row.get('status_decomp', '?'):>16s} "
                f"(tilt={row.get('tilt_decomp', '?'):>6s}°) | "
                f"hull={row.get('status_hull', '?'):>16s} "
                f"(tilt={row.get('tilt_hull', '?'):>6s}°){marker}",
                flush=True,
            )
            results.append(row)

    print("\n\n=== SUMMARY: Objects where convexHull changes stability ===")
    for r in results:
        if r.get("status_decomp") != r.get("status_hull"):
            print(f"  {r['name']:<50s} {r.get('status_decomp', '?'):>16s} -> {r.get('status_hull', '?'):>16s}")

    print("\n=== SUMMARY: Objects unstable even WITH convexHull ===")
    for r in results:
        if r.get("status_hull") not in ("stable", None) and not str(r.get("status_hull", "")).startswith("ERROR"):
            print(
                f"  {r['name']:<50s} hull={r.get('status_hull', '?'):>16s} "
                f"(tilt={r.get('tilt_hull', '?')}°, z_drop={r.get('z_drop_hull', '?')}m)"
            )


if __name__ == "__main__":
    main()
