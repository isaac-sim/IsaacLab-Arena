# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from isaaclab_arena.analysis.sensitivity.pdf_report import generate_pdf_report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-PDF sensitivity report (outcome × factor grid of "
            "marginal-posterior plots) from a (factors.yaml, episode_summary.jsonl) pair."
        )
    )
    parser.add_argument("--factors_yaml", type=str, required=True, help="Path to factors.yaml.")
    parser.add_argument(
        "--episode_summary", type=str, required=True, help="Path to episode_summary.jsonl produced by eval_runner."
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default="./sensitivity_report.pdf",
        help="Output PDF file. Default: ./sensitivity_report.pdf.",
    )
    args = parser.parse_args()

    generate_pdf_report(args.factors_yaml, args.episode_summary, args.output_pdf)


if __name__ == "__main__":
    main()
