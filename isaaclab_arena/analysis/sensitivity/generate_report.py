# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.analysis.sensitivity.plotting import plot_marginals


def generate_report(
    episode_results_path: str | Path,
    output_path: str | Path,
    outcome_names: list[str] | tuple[str, ...] = ("success",),
    observation: list[float] | None = None,
    seed: int | None = 0,
) -> Path:
    """Build a sensitivity report from an episode_results.jsonl, fit, and save a figure.

    The factor schema is discovered from the recorder's per-episode variation draws. The output
    format follows the output_path extension (.png, .pdf, …).

    Args:
        episode_results_path: episode_results.jsonl produced by the per-episode recorder.
        output_path: Destination figure file (parent dirs created if absent).
        outcome_names: Which per-episode outcome(s) to condition on.
        observation: Outcome values to condition on, one per outcome name. Defaults to
            conditioning on success (1) for every (binary) outcome.
        seed: Seed for torch's global RNG, set once before fitting so the estimator training
            and posterior sampling are reproducible. Pass ``None`` to leave the RNG untouched.

    Returns:
        The resolved output path.
    """
    # Estimator training (fit) and posterior sampling both draw from torch's global RNG in
    # sequence, so seeding once here makes the whole report reproducible.
    if seed is not None:
        torch.manual_seed(seed)

    dataset = SensitivityDataset.from_episode_results(Path(episode_results_path), outcome_names)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()

    observation_tensor = (
        dataset.default_observation() if observation is None else torch.tensor(observation, dtype=torch.float32)
    )
    samples = analyzer.sample_posterior(observation_tensor)
    output_path = Path(output_path)
    plot_marginals(samples, dataset, observation_tensor, output_path=str(output_path))
    plt.close("all")
    print(f"[INFO] Wrote report → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a sensitivity report (one posterior-marginal panel per factor) from an "
            "episode_results.jsonl. Output format follows the --output extension."
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
        help="Output figure file; format follows the extension (.png, .pdf, …). Default: eval/sensitivity_report.png.",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        nargs="+",
        default=["success"],
        help="Which per-episode outcome(s) to condition on (top-level field(s) in each row). Default: success.",
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
        "--seed",
        type=int,
        default=0,
        help="Seed for torch's global RNG, so estimator training + sampling are reproducible. Default: 0.",
    )
    args = parser.parse_args()

    generate_report(
        args.episode_results,
        args.output,
        outcome_names=args.outcome,
        observation=args.observation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
