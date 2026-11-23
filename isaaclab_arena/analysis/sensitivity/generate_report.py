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
from isaaclab_arena.analysis.sensitivity.episode_results_reader import dataset_from_episode_results
from isaaclab_arena.analysis.sensitivity.plotting import plot_corner, plot_importance, plot_marginals

# Each report plot, mapped to the renderer that draws it. "marginals" writes the given output_path
# directly; the others write a sibling file suffixed with the plot name.
_PLOT_RENDERERS = {
    "marginals": plot_marginals,
    "importance": plot_importance,
    "corner": plot_corner,
}
_DEFAULT_PLOTS = ("marginals", "importance", "corner")


def generate_report(
    episode_results_path: str | Path,
    output_path: str | Path,
    outcome_names: list[str] | tuple[str, ...] = ("success",),
    factor_names: list[str] | tuple[str, ...] | None = None,
    observation: list[float] | None = None,
    plots: list[str] | tuple[str, ...] = _DEFAULT_PLOTS,
    seed: int | None = 0,
) -> dict[str, Path]:
    """Build a sensitivity report from an episode_results.jsonl, fit, and save the requested figures.

    The factor schema is discovered from the recorder's per-episode variation draws, the posterior
    is fit and sampled once, and each requested plot is rendered from those shared samples. The
    output format follows the output_path extension (.png, .pdf, …).

    Args:
        episode_results_path: episode_results.jsonl produced by the per-episode recorder.
        output_path: Destination figure file for the marginals plot; the other plots are written
            as siblings suffixed with their name (e.g. report_importance.png). Parent dirs created.
        outcome_names: Which per-episode outcome(s) to condition on.
        factor_names: Which recorded variations to analyze. None analyzes all of them.
        observation: Outcome values to condition on, one per outcome name. None conditions on
            success (1) for every binary outcome.
        plots: Which plots to render, any of "marginals", "importance", "corner".
        seed: Seed for torch's global RNG so a report is reproducible. Pass None to leave the
            RNG untouched.

    Returns:
        A mapping of each rendered plot name to its written path.
    """
    unknown = [name for name in plots if name not in _PLOT_RENDERERS]
    assert not unknown, f"Unknown plot(s) {unknown}; choose from {sorted(_PLOT_RENDERERS)}."

    # Estimator training (fit) and posterior sampling both draw from torch's global RNG in
    # sequence, so seeding once here makes the whole report reproducible.
    if seed is not None:
        torch.manual_seed(seed)

    dataset = dataset_from_episode_results(episode_results_path, outcome_names, factor_names)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()

    observation_tensor = (
        dataset.default_observation() if observation is None else torch.tensor(observation, dtype=torch.float32)
    )
    samples = analyzer.sample_posterior(observation_tensor)

    output_path = Path(output_path)
    written: dict[str, Path] = {}
    for name in plots:
        # The marginals plot keeps output_path; the others get a name-suffixed sibling.
        plot_path = output_path if name == "marginals" else output_path.with_stem(f"{output_path.stem}_{name}")
        _PLOT_RENDERERS[name](samples, dataset, observation_tensor, output_path=str(plot_path))
        written[name] = plot_path
    plt.close("all")
    print(f"[INFO] Wrote report → {', '.join(str(path) for path in written.values())}")
    return written


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
        "--plots",
        type=str,
        nargs="+",
        choices=sorted(_PLOT_RENDERERS),
        default=list(_DEFAULT_PLOTS),
        help="Which plots to render. Default: marginals importance corner.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch's global RNG so a report is reproducible. Default: 0.",
    )
    args = parser.parse_args()

    generate_report(
        args.episode_results,
        args.output,
        outcome_names=args.outcome,
        factor_names=args.factors,
        observation=args.observation,
        plots=args.plots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
