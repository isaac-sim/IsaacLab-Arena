# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

INTENSITY_LOW = 10.0
INTENSITY_HIGH = 5000.0

# Inline factors.yaml template (not imported) so this stays a pure-python dev tool —
# importing episode_writer would pull in pxr via isaaclab_arena.metrics.
_SYNTHETIC_FACTORS_YAML = """\
# factors.yaml — synthetic dataset for analyzer smoke-testing.
# Auto-emitted by isaaclab_arena.tests.utils.synthetic_data_continuous alongside the JSONL.

slice:
  policy: synthetic_linear_uniform
  task: synthetic_pick_and_place
  embodiment: synthetic

factors:
  light_intensity:
    type: continuous
    dim: 1

outcomes:
  success_rate:
    type: float
  object_moved_rate:
    type: float
"""


def success_probability(intensity: float, center: float, sigma: float) -> float:
    """Linear-Gaussian competence band: peaks at `center`, falls off symmetrically in linear space."""
    z_score = (intensity - center) / sigma
    return math.exp(-0.5 * z_score * z_score)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic episode_summary.jsonl with a known linear-Gaussian competence band "
            "for smoke-testing the continuous sensitivity analysis pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=str, default="/tmp/synthetic_episode_summary.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--factors-yaml-out",
        type=str,
        default=None,
        help="Output factors.yaml path. Default: same directory as --output, named factors.yaml.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=180,
        help="Total number of episodes to generate. Each draws an intensity from Uniform(10, 5000).",
    )
    parser.add_argument("--center", type=float, default=500.0, help="Intensity where success rate peaks. Default: 500.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=400.0,
        help=(
            "Linear-space width of the competence band (1 sigma in intensity units). Default: 400,"
            " which gives ~95%% success in [100, 900] and near-zero success beyond ~1700."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    random_generator = random.Random(args.seed)

    summary_rows = []
    for episode_index in range(args.num_episodes):
        intensity = random_generator.uniform(INTENSITY_LOW, INTENSITY_HIGH)
        probability_of_success = success_probability(intensity, args.center, args.sigma)
        was_success = 1.0 if random_generator.random() < probability_of_success else 0.0
        summary_rows.append({
            "job_name": "synth_linear_uniform",
            "episode_idx": episode_index,
            "arena_env_args": {"light_intensity": intensity},
            "outcomes": {
                "success_rate": was_success,
                "object_moved_rate": was_success,
            },
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for summary_row in summary_rows:
            jsonl_file.write(json.dumps(summary_row) + "\n")

    # Emit a matching factors.yaml so the analyzer can be pointed at this synthetic dataset
    # without any hand-authored schema. Inline string template — see _SYNTHETIC_FACTORS_YAML.
    factors_yaml_path = Path(args.factors_yaml_out) if args.factors_yaml_out else output_path.parent / "factors.yaml"
    factors_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    factors_yaml_path.write_text(_SYNTHETIC_FACTORS_YAML, encoding="utf-8")

    print(f"[INFO] Wrote {len(summary_rows)} rows to {output_path}")
    print(f"[INFO] Wrote factors schema → {factors_yaml_path}")
    print(f"[INFO] Linear-Gaussian competence band: center={args.center:g}, sigma={args.sigma:g}")
    print("[INFO] Per-bin success rates (10 equal bins across the prior range):")
    num_bins = 10
    bin_width = (INTENSITY_HIGH - INTENSITY_LOW) / num_bins
    for bin_index in range(num_bins):
        bin_low = INTENSITY_LOW + bin_index * bin_width
        bin_high = bin_low + bin_width
        rows_in_bin = [row for row in summary_rows if bin_low <= row["arena_env_args"]["light_intensity"] < bin_high]
        if not rows_in_bin:
            continue
        successes_in_bin = sum(int(row["outcomes"]["success_rate"]) for row in rows_in_bin)
        percentage = 100 * successes_in_bin / len(rows_in_bin)
        bar_string = "█" * int(round(percentage / 5))
        print(
            f"       [{bin_low:>5g}, {bin_high:>5g}): {successes_in_bin:>3d}/{len(rows_in_bin):<3d}"
            f" ({percentage:>5.1f}%) {bar_string}"
        )


if __name__ == "__main__":
    main()
