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
    factors_yaml_path: str | Path,
    jsonl_path: str | Path,
    output_path: str | Path,
    observation: list[float] | None = None,
) -> Path:
    """Build a sensitivity report from a factors.yaml / episode_summary.jsonl pair.

    Loads the data, fits a SensitivityAnalyzer, and saves a single posterior-marginals
    figure. The output format follows the output_path extension (.png, .pdf, …).

    Args:
        factors_yaml_path: Schema file declaring factors and outcomes.
        jsonl_path: episode_summary.jsonl produced by eval_runner.
        output_path: Destination figure file (parent dirs created if absent).
        observation: Outcome values to condition on, one per declared outcome. Defaults to
            conditioning on success (1) for every binary outcome.

    Returns:
        The resolved output path.
    """
    dataset = SensitivityDataset.from_files(Path(factors_yaml_path), Path(jsonl_path))
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
        default="eval/sensitivity_report.png",
        help="Output figure file; format follows the extension (.png, .pdf, …). Default: eval/sensitivity_report.png.",
    )
    parser.add_argument(
        "--observation",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Outcome values to condition on, one per declared outcome (in schema order). "
            "Outcomes are binary, so use 1 for success or 0 for failure. Defaults to 1 (success)."
        ),
    )
    args = parser.parse_args()

    generate_report(args.factors_yaml, args.episode_summary, args.output, observation=args.observation)


if __name__ == "__main__":
    main()
