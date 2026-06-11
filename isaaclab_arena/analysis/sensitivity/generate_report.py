# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from isaaclab_arena.analysis.sensitivity.report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a sensitivity report (one posterior-marginal panel per factor) from a "
            "(factors.yaml, episode_summary.jsonl) pair. Output format follows the --output extension."
        )
    )
    parser.add_argument("--factors_yaml", type=str, required=True, help="Path to factors.yaml.")
    parser.add_argument(
        "--episode_summary", type=str, required=True, help="Path to episode_summary.jsonl produced by eval_runner."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sensitivity_report.png",
        help="Output figure file; format follows the extension (.png, .pdf, …). Default: ./sensitivity_report.png.",
    )
    parser.add_argument(
        "--observation",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Outcome values to condition on, one per declared outcome (in schema order). "
            "Defaults to 1.0 for binary outcomes and the mean otherwise."
        ),
    )
    args = parser.parse_args()

    generate_report(args.factors_yaml, args.episode_summary, args.output, observation=args.observation)


if __name__ == "__main__":
    main()
