# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from isaaclab_arena.analysis.sensitivity import make_analyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.analysis.sensitivity.plotting import plot_marginal


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline 1D continuous sensitivity analysis: fit an analyzer on a "
            "(factors.yaml, episode_summary.jsonl) pair and save a posterior-marginal plot."
        )
    )
    parser.add_argument("--factors_yaml", type=str, required=True, help="Path to factors.yaml.")
    parser.add_argument(
        "--episode_summary", type=str, required=True, help="Path to episode_summary.jsonl produced by eval_runner."
    )
    parser.add_argument(
        "--input_factor",
        type=str,
        default=None,
        help="Name of the factor to plot. Defaults to the only factor declared in factors.yaml.",
    )
    parser.add_argument(
        "--output_metric",
        type=str,
        default=None,
        help="Outcome name to condition on. Defaults to the first outcome listed in factors.yaml.",
    )
    parser.add_argument(
        "--outcome_value",
        type=float,
        default=1.0,
        help="Outcome value to condition on (1.0 = success). Default: 1.0.",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default="./sensitivity.png",
        help="Output figure path. Default: ./sensitivity.png.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading dataset: factors={args.factors_yaml}  jsonl={args.episode_summary}")
    dataset = SensitivityDataset(args.factors_yaml, args.episode_summary)

    available_factors = list(dataset.factor_columns)
    available_outcomes = [outcome.name for outcome in dataset.schema.outcomes]

    if args.input_factor is None:
        factor_name = available_factors[0]
    else:
        if args.input_factor not in available_factors:
            parser.error(
                f"--input_factor {args.input_factor!r} not found in factors.yaml. "
                f"Available factors: {available_factors}"
            )
        factor_name = args.input_factor

    if args.output_metric is None:
        outcome_name = available_outcomes[0]
    else:
        if args.output_metric not in available_outcomes:
            parser.error(
                f"--output_metric {args.output_metric!r} not found in factors.yaml. "
                f"Available outcomes: {available_outcomes}"
            )
        outcome_name = args.output_metric

    print(
        f"[INFO] Analyzing factor '{factor_name}' against outcome '{outcome_name}'"
        f" (conditioning on outcome={args.outcome_value:g})"
    )
    print(
        f"[INFO] num_episodes={len(dataset.rows)};  theta shape={tuple(dataset.theta.shape)};"
        f"  x shape={tuple(dataset.x.shape)}"
    )

    analyzer = make_analyzer(dataset, outcome_name=outcome_name)
    print(f"[INFO] Dispatched analyzer: {type(analyzer).__name__}")
    analyzer.fit()
    print(f"[INFO] Plotting marginal -> {args.figure_path}")
    plot_marginal(analyzer, factor_name, output_path=args.figure_path, outcome_value=args.outcome_value)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
