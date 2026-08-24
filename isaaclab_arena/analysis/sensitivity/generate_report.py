# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Literal

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.empirical import compute_empirical_marginals
from isaaclab_arena.analysis.sensitivity.episode_results_reader import dataset_from_episode_results
from isaaclab_arena.analysis.sensitivity.plotting import plot_empirical_marginals, plot_marginals


def generate_report(
    episode_results_path: str | Path,
    output_path: str | Path,
    outcome_names: list[str] | tuple[str, ...] = ("success",),
    factor_names: list[str] | tuple[str, ...] | None = None,
    observation: list[float] | None = None,
    seed: int | None = 0,
    method: Literal["fitted", "empirical"] = "fitted",
    num_bins: int = 6,
    num_bootstrap_samples: int = 1000,
) -> Path:
    """Build and save a fitted or empirical sensitivity report from episode results.

    The factor schema is discovered from the recorder's per-episode variation draws. The output
    format follows the output_path extension (.png, .pdf, …).

    Args:
        episode_results_path: episode_results.jsonl produced by the per-episode recorder.
        output_path: Destination figure file (parent dirs created if absent).
        outcome_names: Which per-episode outcome(s) to condition on.
        factor_names: Which recorded variations to analyze. None analyzes all of them.
        observation: Outcome values to condition on, one per outcome name. None conditions on
            success (1) for every binary outcome.
        seed: Seed for fitted inference or empirical bootstrap resampling. Pass None for
            nondeterministic behavior.
        method: ``fitted`` trains NPE or MNPE; ``empirical`` uses exactly matching episodes.
        num_bins: Number of fixed bins per continuous factor in an empirical report.
        num_bootstrap_samples: Number of paired episode resamples used for empirical intervals.

    Returns:
        The resolved output path.
    """
    assert method in {"fitted", "empirical"}, f"Unknown sensitivity method {method!r}."
    dataset = dataset_from_episode_results(episode_results_path, outcome_names, factor_names)
    observation_tensor = (
        dataset.default_observation() if observation is None else torch.tensor(observation, dtype=torch.float32)
    )
    output_path = Path(output_path)

    if method == "empirical":
        empirical_result = compute_empirical_marginals(
            dataset,
            observation_tensor,
            num_bins=num_bins,
            num_bootstrap_samples=num_bootstrap_samples,
            seed=seed,
        )
        print(
            f"[INFO] Empirical sensitivity: {empirical_result.num_matching_episodes} of "
            f"{empirical_result.num_episodes} episodes exactly match the observation."
        )
        plot_empirical_marginals(empirical_result, dataset, output_path=str(output_path))
    else:
        # Estimator training and posterior sampling both draw from torch's global RNG in
        # sequence, so seeding once here makes the fitted report reproducible.
        if seed is not None:
            torch.manual_seed(seed)
        analyzer = SensitivityAnalyzer(dataset)
        analyzer.fit()
        samples = analyzer.sample_posterior(observation_tensor)
        plot_marginals(samples, dataset, observation_tensor, output_path=str(output_path))

    plt.close("all")
    print(f"[INFO] Wrote report → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a fitted or empirical sensitivity report from an episode_results.jsonl. "
            "Output format follows the --output extension."
        )
    )
    parser.add_argument(
        "--episode_results",
        type=str,
        required=True,
        help="Path to episode_results.jsonl produced by the per-episode recorder.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/sensitivity_report.png",
        help="Output figure file. Format follows the extension (.png, .pdf, …). Default: eval/sensitivity_report.png.",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        nargs="+",
        default=["success"],
        help="Which per-episode outcome(s) to condition on (top-level field(s) in each row). Default: success.",
    )
    parser.add_argument(
        "--factors",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Which recorded variations to analyze (keys in each row's variations block, a vector "
            "variation keeps all its components). Default: all recorded variations."
        ),
    )
    parser.add_argument(
        "--observation",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Outcome values to condition on, one per --outcome (in order). "
            "Outcomes are binary, so use 1 for success or 0 for failure. Defaults to 1 (success)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["fitted", "empirical"],
        default="fitted",
        help=(
            "Posterior method. 'empirical' uses episodes that exactly match the observation; "
            "'fitted' trains NPE or MNPE. Default: fitted."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=6,
        help="Number of fixed bins per continuous factor for --method empirical. Default: 6.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        "--bootstrap-samples",
        dest="bootstrap_samples",
        type=int,
        default=1000,
        help="Number of paired episode resamples for empirical confidence intervals. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for fitted inference or empirical bootstrap resampling. Default: 0.",
    )
    args = parser.parse_args()

    generate_report(
        args.episode_results,
        args.output,
        outcome_names=args.outcome,
        factor_names=args.factors,
        observation=args.observation,
        seed=args.seed,
        method=args.method,
        num_bins=args.bins,
        num_bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
