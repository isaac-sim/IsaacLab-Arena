# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive sensitivity explorer: a Streamlit shell over the amortized posterior.

The estimator is amortized, so re-conditioning it on a new outcome is a cheap re-sample with no
retraining. This app exposes that: the posterior is fit once (cached), and every control change
re-samples it and redraws the importance ranking, the per-factor marginals, and a chosen pairwise
joint. It is a development/exploration tool — Streamlit is a dev dependency, not a runtime one.

Run (after ``pip install -e .[dev]`` for streamlit):

    streamlit run isaaclab_arena/analysis/sensitivity/app.py -- --episode_results path/to/episode_results.jsonl

Everything after the ``--`` is passed to this script; the path can also be set in the sidebar.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch

import streamlit as st

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.analysis.sensitivity.episode_results_reader import dataset_from_episode_results
from isaaclab_arena.analysis.sensitivity.marginals import condition_mask
from isaaclab_arena.analysis.sensitivity.plotting import plot_marginal

# Posterior sampling is amortized (cheap), so we draw a large fixed pool: enough to keep conditioned
# slices (the what-if panel) populated without a user-facing knob. It only sets MC resolution.
_NUM_SAMPLES = 50000
_THIN_SLICE_WARNING = 200
"""Below this many draws in a conditioned slice, warn that the curve is unreliable."""


def _parse_args() -> argparse.Namespace:
    """Parse the CLI args Streamlit forwards after ``--`` (path is optional; sidebar can set it)."""
    parser = argparse.ArgumentParser(description="Interactive sensitivity explorer.")
    parser.add_argument("--episode_results", type=str, default="", help="Path to episode_results.jsonl.")
    parser.add_argument("--outcome", type=str, nargs="+", default=["success"], help="Outcome field(s).")
    # Streamlit injects its own argv; parse_known_args ignores anything that isn't ours.
    args, _ = parser.parse_known_args()
    return args


@st.cache_resource(show_spinner="Fitting posterior…")
def _load_and_fit(
    episode_results_path: str, outcome_names: tuple[str, ...], seed: int
) -> tuple[SensitivityDataset, SensitivityAnalyzer]:
    """Build the dataset and fit the analyzer once per (path, outcomes, seed).

    Cached as a resource: the fitted analyzer holds a torch model, so it is reused across reruns
    and only refit when one of these inputs changes.
    """
    torch.manual_seed(seed)
    dataset = dataset_from_episode_results(episode_results_path, outcome_names)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()
    return dataset, analyzer


def _outcome_controls(dataset: SensitivityDataset) -> torch.Tensor:
    """Render one sidebar control per outcome and return the observation vector to condition on.

    A binary outcome (values all 0/1) gets a success/failure toggle; any other outcome gets a
    slider over its observed range — the continuous-conditioning the amortized posterior allows.
    """
    st.sidebar.subheader("Condition on outcome")
    values: list[float] = []
    for index, name in enumerate(dataset.outcome_names):
        column = dataset.x[:, index]
        is_binary = set(column.tolist()).issubset({0.0, 1.0})
        if is_binary:
            choice = st.sidebar.radio(
                name, options=[1.0, 0.0], format_func=lambda v: "success (1)" if v == 1.0 else "failure (0)"
            )
            values.append(float(choice))
        else:
            low, high = float(column.min()), float(column.max())
            values.append(st.sidebar.slider(name, min_value=low, max_value=high, value=high))
    return torch.tensor(values, dtype=torch.float32)


def _conditioning_panel(samples: torch.Tensor, dataset: SensitivityDataset, observation: torch.Tensor) -> None:
    """What-if panel: pin some factors and view another's conditional posterior marginal.

    Picks a factor to view, lets every other factor be pinned (continuous → a range band, categorical
    → a choice), slices the draws to that pinned region, and redraws the view factor's marginal from
    the survivors. Pinned factors are conditioned on; unpinned ones are averaged over. A live count
    surfaces when the slice is too thin to trust.
    """
    factor_names = [factor.name for factor in dataset.factors]
    if not factor_names:
        return

    st.subheader("Conditioning (what-if)")
    st.caption("Pin other factors to slice the posterior; unpinned factors are averaged over.")
    view = st.selectbox("View factor", factor_names, index=0, key="condition_view")

    continuous_windows: dict[str, tuple[float, float]] = {}
    categorical_choices: dict[str, int] = {}
    for factor in dataset.factors:
        if factor.name == view:
            continue
        if not st.checkbox(f"pin {factor.name}", value=False, key=f"pin_{factor.name}"):
            continue
        if factor.type == "continuous":
            low, high = float(factor.range[0]), float(factor.range[1])
            span = high - low
            # Default to a central band so pinning has a visible effect; the user widens/moves it.
            window = st.slider(
                factor.name,
                min_value=low,
                max_value=high,
                value=(low + 0.4 * span, low + 0.6 * span),
                key=f"window_{factor.name}",
            )
            continuous_windows[factor.name] = window
        else:
            choice = st.selectbox(f"{factor.name} =", factor.choices, key=f"choice_{factor.name}")
            categorical_choices[factor.name] = factor.choices.index(choice)

    mask = condition_mask(samples, dataset, continuous_windows, categorical_choices)
    num_in_slice = int(mask.sum())
    st.caption(f"{num_in_slice} / {len(mask)} samples in slice")
    if num_in_slice == 0:
        st.warning("No samples in this slice — widen a window or unpin a factor.")
        return
    if num_in_slice < _THIN_SLICE_WARNING:
        st.warning(
            f"Thin slice ({num_in_slice} samples): the curve reflects the fitted model more than the "
            "data here. Widen a window or unpin a factor."
        )
    st.pyplot(plot_marginal(samples[torch.as_tensor(mask)], dataset, view, observation), use_container_width=False)


def main() -> None:
    """Run the interactive explorer: fit once, then re-sample and redraw on every control change."""
    st.set_page_config(page_title="Sensitivity Explorer", layout="wide")
    st.title("Sensitivity Explorer")

    args = _parse_args()
    episode_results_path = st.sidebar.text_input("episode_results.jsonl", value=args.episode_results)
    if not episode_results_path:
        st.info("Set the path to an episode_results.jsonl in the sidebar to begin.")
        return

    seed = st.sidebar.number_input("seed", value=0, step=1)
    dataset, analyzer = _load_and_fit(episode_results_path, tuple(args.outcome), int(seed))

    observation = _outcome_controls(dataset)

    # Re-seed before sampling so identical controls reproduce the same draws across reruns.
    torch.manual_seed(int(seed))
    samples = analyzer.sample_posterior(observation, num_samples=_NUM_SAMPLES)

    _conditioning_panel(samples, dataset, observation)

    plt.close("all")


if __name__ == "__main__":
    main()
