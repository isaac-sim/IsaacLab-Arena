# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI driver: build a static HTML sensitivity report from a (factors.yaml, JSONL) pair.

Thin wrapper around :func:`isaaclab_arena.analysis.sensitivity.report.generate_report`.
For per-plot inspection or A/B comparisons of a single factor/outcome, use
``analyze_sensitivity.py`` instead — this script produces the full deliverable artifact
covering every declared (factor, outcome) combination in one self-contained HTML file.

Example::

    python -m isaaclab_arena.scripts.generate_sensitivity_report \\
        --factors_yaml path/to/factors.yaml \\
        --episode_summary path/to/episode_summary.jsonl \\
        --output_html /tmp/sensitivity_report.html
"""

from __future__ import annotations

import argparse

from isaaclab_arena.analysis.sensitivity.report import generate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factors_yaml", type=str, required=True, help="Path to factors.yaml.")
    parser.add_argument(
        "--episode_summary", type=str, required=True, help="Path to episode_summary.jsonl produced by eval_runner."
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="./sensitivity_report.html",
        help="Output HTML file. Default: ./sensitivity_report.html. Self-contained interactive HTML.",
    )
    parser.add_argument(
        "--plotlyjs_mode",
        choices=["cdn", "inline"],
        default="cdn",
        help=(
            "Plotly.js bundling. 'cdn' (default, ~500 KB output, needs internet to load) loads"
            " Plotly from plot.ly's CDN; 'inline' (~5 MB output, fully offline) embeds Plotly"
            " directly. Use 'inline' when sharing to people who may not have internet."
        ),
    )
    args = parser.parse_args()

    generate_report(args.factors_yaml, args.episode_summary, args.output_html, plotlyjs_mode=args.plotlyjs_mode)


if __name__ == "__main__":
    main()
