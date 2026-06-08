# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Five objects like the maple-table sweep: first three "easy" (high success), last two
# "hard" (low) — a known signal the analyzer should recover.
DEFAULT_CHOICES = [
    "rubiks_cube_hot3d_robolab",
    "wooden_bowl_hot3d_robolab",
    "alphabet_soup_can_hope_robolab",
    "mug_ycb_robolab",
    "sugar_box_ycb_robolab",
]
DEFAULT_SUCCESS_PROBABILITIES = [0.90, 0.85, 0.75, 0.25, 0.15]


def _factors_yaml_text(choices: list[str]) -> str:
    """Build the factors.yaml content matching the synthetic data."""
    choices_string = ", ".join(choices)
    return (
        "# factors.yaml — synthetic categorical dataset for analyzer smoke-testing.\n"
        "# Auto-emitted by synthetic_data_categorical alongside the JSONL.\n"
        "\n"
        "slice:\n"
        "  policy: synthetic_categorical\n"
        "  task: synthetic_pick_and_place\n"
        "  embodiment: synthetic\n"
        "\n"
        "factors:\n"
        "  pick_up_object:\n"
        "    type: categorical\n"
        f"    choices: [{choices_string}]\n"
        "\n"
        "outcomes:\n"
        "  success_rate:\n"
        "    type: float\n"
        "  object_moved_rate:\n"
        "    type: float\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic episode_summary.jsonl where a single categorical factor drives the "
            "success probability, for smoke-testing the categorical sensitivity analysis pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/synthetic_categorical_episode_summary.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--factors-yaml-out",
        type=str,
        default=None,
        help="Output factors.yaml path. Default: same directory as --output, named factors.yaml.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=200,
        help="Total episodes (uniform draws across all choices). Default 200 → ~40 per category for 5 choices.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    random_generator = random.Random(args.seed)
    choices = DEFAULT_CHOICES
    success_probabilities = DEFAULT_SUCCESS_PROBABILITIES
    assert len(choices) == len(
        success_probabilities
    ), "DEFAULT_CHOICES and DEFAULT_SUCCESS_PROBABILITIES lengths must match"
    num_choices = len(choices)

    summary_rows = []
    per_category_stats: dict[str, list[int]] = {choice: [0, 0] for choice in choices}  # category → [successes, total]
    for episode_index in range(args.num_episodes):
        category_index = random_generator.randrange(num_choices)
        chosen_category = choices[category_index]
        was_success = 1.0 if random_generator.random() < success_probabilities[category_index] else 0.0
        per_category_stats[chosen_category][0] += int(was_success)
        per_category_stats[chosen_category][1] += 1
        summary_rows.append({
            "job_name": "synth_categorical",
            "episode_idx": episode_index,
            "arena_env_args": {"pick_up_object": chosen_category},
            "outcomes": {"success_rate": was_success, "object_moved_rate": was_success},
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for summary_row in summary_rows:
            jsonl_file.write(json.dumps(summary_row) + "\n")

    factors_yaml_path = Path(args.factors_yaml_out) if args.factors_yaml_out else output_path.parent / "factors.yaml"
    factors_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    factors_yaml_path.write_text(_factors_yaml_text(choices), encoding="utf-8")

    print(f"[INFO] Wrote {len(summary_rows)} rows to {output_path}")
    print(f"[INFO] Wrote factors schema → {factors_yaml_path}")
    print("[INFO] Per-category success counts (analyzer should pull posterior mass toward easy cats):")
    for choice, target_probability in zip(choices, success_probabilities):
        successes, total = per_category_stats[choice]
        empirical_percentage = 100 * successes / total if total else 0.0
        bar_string = "█" * int(round(empirical_percentage / 5))
        print(
            f"       {choice:<35s} target={target_probability:>4.0%}"
            f"  empirical={successes:>3d}/{total:<3d} ({empirical_percentage:>5.1f}%) {bar_string}"
        )


if __name__ == "__main__":
    main()
