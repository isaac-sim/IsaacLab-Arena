# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI driver: build a single-PDF sensitivity report from a (factors.yaml, JSONL) pair.

Thin wrapper around :func:`isaaclab_arena.analysis.sensitivity.pdf_report.generate_pdf_report`.
Produces one PDF with an outcome × factor grid of marginal-posterior plots — the most
important plots in a single file. For per-plot inspection of a single factor/outcome, use
``analyze_sensitivity.py`` instead.

Example::

    python -m isaaclab_arena.scripts.generate_sensitivity_report \\
        --factors_yaml path/to/factors.yaml \\
        --episode_summary path/to/episode_summary.jsonl \\
        --output_pdf /tmp/sensitivity_report.pdf
"""

from __future__ import annotations

import argparse

from isaaclab_arena.analysis.sensitivity.pdf_report import generate_pdf_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
