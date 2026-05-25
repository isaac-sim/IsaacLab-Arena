# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scan robolab objects for stability under natural placement (On table, solver-placed).

Each object is tested in its own subprocess to avoid SimApp reuse crashes.
Runs ``run_stability_check.py --json`` per object, parses the JSON output,
and prints a comparison table.

Usage (inside container)::

    # Scan all robolab objects (no convexHull fix — baseline):
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py

    # Scan with convexHull fix enabled:
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --force_convex_hull

    # Compare both modes side by side:
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --compare

    # Test specific objects only:
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py \
        --objects mustard_bottle_hope_robolab milk_carton_hope_robolab

    # Multi-object scene (4 objects together):
    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --multi
"""

from __future__ import annotations

import json
import math
import os
import subprocess

PYTHON = os.environ.get("ISAAC_SIM_PYTHON", "/isaac-sim/python.sh")
CHECKER = "isaaclab_arena/llm_env_gen/run_stability_check.py"

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

MULTI_OBJECT_SET = [
    "mustard_bottle_hope_robolab",
    "milk_carton_hope_robolab",
    "alphabet_soup_can_hope_robolab",
    "sugar_box_ycb_robolab",
]


def run_check(object_names: list[str], force_convex_hull: bool = False, timeout: int = 120) -> dict | None:
    """Run stability check in a subprocess and return parsed JSON result."""
    cmd = [
        PYTHON,
        CHECKER,
        "--headless",
        "--json",
        "--num_envs",
        "1",
        "--settle_steps",
        "60",
        "gr1_table_multi_object_no_collision",
        "--embodiment",
        "gr1_joint",
        "--objects",
    ] + object_names

    env = os.environ.copy()
    if force_convex_hull:
        env["ARENA_FORCE_CONVEX_HULL"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode not in (0, 4, 5):
        return None

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def extract_metrics(data: dict, name: str) -> dict | None:
    if data is None:
        return None
    objects = data.get("objects", {})
    return objects.get(name)


def fmt_status(metrics: dict | None) -> tuple[str, float]:
    if metrics is None:
        return "ERROR", 0.0
    return metrics["status"], math.degrees(metrics["tilt_rad"])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scan robolab objects for stability")
    parser.add_argument(
        "--objects", nargs="*", type=str, default=None, help="Specific objects to test. Default: all robolab objects."
    )
    parser.add_argument(
        "--force_convex_hull", action="store_true", default=False, help="Enable convexHull override for all tests."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Run both with and without convexHull and show side-by-side comparison.",
    )
    parser.add_argument("--multi", action="store_true", default=False, help="Also test a multi-object scene.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-object subprocess timeout in seconds.")
    args = parser.parse_args()

    object_list = args.objects if args.objects else ROBOLAB_OBJECTS
    total = len(object_list)

    if args.compare:
        print(f"\n{'=' * 100}")
        print(f"{'ROBOLAB STABILITY SCAN — convexDecomposition vs convexHull':^100}")
        print(f"{'=' * 100}")
        print(
            f"{'#':>3} {'Object':<45s} {'Decomp Status':>14s} {'Tilt':>6s} {'Hull Status':>14s} {'Tilt':>6s} {'Effect':>10s}"
        )
        print(f"{'-' * 100}")

        changed = []
        for i, name in enumerate(object_list):
            decomp_data = run_check([name], force_convex_hull=False, timeout=args.timeout)
            hull_data = run_check([name], force_convex_hull=True, timeout=args.timeout)

            d_status, d_tilt = fmt_status(extract_metrics(decomp_data, name))
            h_status, h_tilt = fmt_status(extract_metrics(hull_data, name))

            effect = ""
            if d_status != h_status and h_status == "stable":
                effect = "FIXED"
                changed.append(name)
            elif d_status != h_status:
                effect = "CHANGED"
                changed.append(name)
            elif d_tilt > 5 and h_tilt < d_tilt * 0.7:
                effect = "improved"

            print(
                f"{i + 1:>3} {name:<45s} {d_status:>14s} {d_tilt:5.1f}° {h_status:>14s} {h_tilt:5.1f}° {effect:>10s}",
                flush=True,
            )

        print(f"\n{'=' * 100}")
        print(f"SUMMARY: {len(changed)}/{total} objects changed with convexHull")
        if changed:
            for n in changed:
                print(f"  - {n}")
        print(f"{'=' * 100}")

    else:
        hull = args.force_convex_hull
        mode_label = "convexHull" if hull else "convexDecomposition (baseline)"
        print(f"\n{'=' * 90}")
        print(f"{'ROBOLAB STABILITY SCAN — ' + mode_label:^90}")
        print(f"{'=' * 90}")
        print(f"{'#':>3} {'Object':<45s} {'Status':>14s} {'Tilt':>7s} {'Z-drop':>8s} {'XY-drift':>9s} {'Jump1':>7s}")
        print(f"{'-' * 90}")

        unstable = []
        errors = []
        for i, name in enumerate(object_list):
            data = run_check([name], force_convex_hull=hull, timeout=args.timeout)
            m = extract_metrics(data, name)

            if m is None:
                print(f"{i + 1:>3} {name:<45s} {'ERROR':>14s}", flush=True)
                errors.append(name)
                continue

            status = m["status"]
            tilt = math.degrees(m["tilt_rad"])
            z_drop = m["z_drop_m"]
            xy_drift = m["xy_drift_m"]
            jump1 = m["first_step_jump_m"]

            marker = " ***" if status != "stable" else ""
            print(
                f"{i + 1:>3} {name:<45s} {status:>14s} {tilt:5.1f}° {z_drop:7.4f}m {xy_drift:8.4f}m"
                f" {jump1:6.4f}m{marker}",
                flush=True,
            )

            if status != "stable":
                unstable.append((name, status, tilt))

        print(f"\n{'=' * 90}")
        print(f"RESULTS: {total - len(unstable) - len(errors)} stable, {len(unstable)} unstable, {len(errors)} errors")
        if unstable:
            print("\nUnstable objects:")
            for name, status, tilt in unstable:
                print(f"  {name:<45s} {status:>14s} (tilt={tilt:.1f}°)")
        if errors:
            print(f"\nErrors: {', '.join(errors)}")
        print(f"{'=' * 90}")

    if args.multi:
        print(f"\n{'=' * 90}")
        print(f"{'MULTI-OBJECT SCENE':^90}")
        print(f"Objects: {', '.join(MULTI_OBJECT_SET)}")
        print(f"{'=' * 90}")

        for hull, label in [(False, "convexDecomposition"), (True, "convexHull")]:
            data = run_check(MULTI_OBJECT_SET, force_convex_hull=hull, timeout=args.timeout)
            print(f"\n  {label}:")
            if data is None:
                print("    ERROR: subprocess failed")
                continue
            for name in MULTI_OBJECT_SET:
                m = extract_metrics(data, name)
                if m is None:
                    print(f"    {name:<40s} ERROR")
                    continue
                tilt = math.degrees(m["tilt_rad"])
                print(
                    f"    {name:<40s} {m['status']:>12s} "
                    f"tilt={tilt:5.1f}° z_drop={m['z_drop_m']:.3f}m xy_drift={m['xy_drift_m']:.3f}m"
                )


if __name__ == "__main__":
    main()
