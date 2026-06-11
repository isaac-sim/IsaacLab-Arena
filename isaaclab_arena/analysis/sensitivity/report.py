# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            the analyzer's default (1.0 for binary outcomes, the mean otherwise).

    Returns:
        The resolved output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = SensitivityDataset.from_files(Path(factors_yaml_path), Path(jsonl_path))
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()

    observation_tensor = None if observation is None else torch.tensor(observation, dtype=torch.float32)
    output_path = Path(output_path)
    plot_marginals(analyzer, observation=observation_tensor, output_path=str(output_path))
    plt.close("all")
    print(f"[INFO] Wrote report → {output_path}")
    return output_path
