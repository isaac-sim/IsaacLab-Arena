# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import torch
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FactorType(str, Enum):
    """Whether a factor's values are continuous (numeric range) or categorical (labelled choices)."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class FactorSpec:
    """One factor's schema, occupying a single column of theta.

    A continuous factor carries a range, one [low, high] pair. A categorical factor carries
    choices, a list of string labels that are integer-encoded by their index in theta.
    """

    name: str
    type: FactorType
    range: list[tuple[float, float]] | None = None  # a single (low, high) pair, continuous only
    choices: list[str] | None = None  # categorical only

    def __post_init__(self) -> None:
        # Accept the raw string form (from YAML / callers) and normalize to the enum.
        self.type = FactorType(self.type)
        # Normalize each (low, high) pair to a tuple (YAML/JSON deliver them as lists).
        if self.range is not None:
            self.range = [tuple(pair) for pair in self.range]


class SensitivityDataset:
    """The varied factors paired with their per-episode theta (factor values) and x (outcomes).

    A pure container holding the factor list, the two tensors, and the column layout an analyzer
    reads. Build it from an episode_results.jsonl with from_episode_results, or pass tensors
    straight to the constructor. Either way theta is continuous columns first, then one
    integer-coded column per categorical factor.
    """

    def __init__(
        self,
        factors: list[FactorSpec],
        theta: torch.Tensor,
        x: torch.Tensor,
        outcome_names: list[str] | tuple[str, ...] = ("success",),
    ):
        """Wrap an in-memory factor list plus its theta / x tensors, validating shapes.

        Args:
            factors: The varied factors, one per theta column. A continuous factor must carry a
                range, a categorical factor must carry choices.
            theta: (num_episodes, num_factors) factor matrix, continuous-first.
            x: (num_episodes, num_outcomes) outcome matrix.
            outcome_names: Name of each outcome column in x, in order (used for plot labels).
        """
        assert theta.ndim == 2 and x.ndim == 2, f"theta and x must be 2D; got {theta.shape} and {x.shape}"
        assert theta.shape[0] == x.shape[0], f"theta/x row counts disagree: {theta.shape[0]} vs {x.shape[0]}"
        assert theta.shape[0] > 0, "Dataset is empty (no episodes)"
        assert theta.shape[1] == len(
            factors
        ), f"theta has {theta.shape[1]} columns but there are {len(factors)} factor(s) (one column each)"
        assert x.shape[1] == len(
            outcome_names
        ), f"x has {x.shape[1]} columns but {len(outcome_names)} outcome name(s) were given"
        self.factors = factors
        self.outcome_names = list(outcome_names)
        self._theta = theta
        self._x = x

    @classmethod
    def from_episode_results(
        cls,
        jsonl_path: str | Path,
        outcome_names: list[str] | tuple[str, ...] = ("success",),
        factor_names: list[str] | tuple[str, ...] | None = None,
    ) -> SensitivityDataset:
        """Build a dataset from an episode_results.jsonl, discovering the factors from the data.

        Each line is one episode. The variations block holds the sampled factor draws, and the
        top-level fields named by outcome_names hold the outcomes. Other top-level fields are
        ignored. A number becomes a continuous factor, a numeric vector becomes one continuous
        factor per component (named key[i]), and a string becomes a categorical factor over its
        observed labels.

        Example line, one vector and one string factor:

            {"success": true,
             "variations": {"wrist_camera": [0.01, -0.02, 0.0], "hdr_image": "sunset"}}

        Args:
            jsonl_path: Path to the episode_results.jsonl, one JSON object per line.
            outcome_names: Top-level field(s) per line to use as outcomes.
            factor_names: Which recorded variations to analyze, by their variations-block name. A
                vector is selected by its base name and keeps every component. None analyzes all.

        Returns:
            A SensitivityDataset whose theta / x use the continuous-first layout the analyzers read.
        """
        jsonl_text = Path(jsonl_path).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
        assert len(rows) > 0, f"Empty episode_results.jsonl at {jsonl_path}"

        factors, theta, x = _build_dataset_from_episode_rows(rows, outcome_names, jsonl_path, factor_names)
        return cls(factors, theta, x, outcome_names)

    @property
    def theta(self) -> torch.Tensor:
        """(num_episodes, num_factors) matrix of factor values, one row per episode.

        The column layout is given by factor_columns, continuous factors first then categoricals
        (integer-coded).
        """
        return self._theta

    @property
    def x(self) -> torch.Tensor:
        """(num_episodes, num_outcomes) matrix of outcome values, one row per episode.

        Columns are named by outcome_names. These are the values a query conditions on.
        """
        return self._x

    @property
    def num_episodes(self) -> int:
        """Number of episodes (rows) in the dataset."""
        return self._theta.shape[0]

    @property
    def factor_columns(self) -> dict[str, slice]:
        """Map each factor name to its single-column slice in theta.

        Continuous factors take the leading columns, then categoricals. Each factor is one column.
        """
        continuous = [factor for factor in self.factors if factor.type == "continuous"]
        categorical = [factor for factor in self.factors if factor.type == "categorical"]
        return {factor.name: slice(index, index + 1) for index, factor in enumerate(continuous + categorical)}

    def default_observation(self) -> torch.Tensor:
        """The outcome vector a query conditions on by default: success (1) for every outcome.

        Outcomes are binary (0/1), so the natural query is what produced success. The assertion
        keeps a continuous outcome from being used here silently.
        """
        is_binary = set(self._x.flatten().tolist()).issubset({0.0, 1.0})
        assert is_binary, "default_observation assumes binary (0/1) outcomes; pass an explicit observation otherwise."
        return torch.ones(self._x.shape[1], dtype=torch.float32)

    @property
    def has_categorical_factors(self) -> bool:
        """True iff at least one factor is categorical."""
        return any(factor.type == "categorical" for factor in self.factors)


def _flatten_variation_value(
    key: str, value: Any, row_index: int, jsonl_path: str | Path
) -> list[tuple[str, float | str]]:
    """Turn one recorded variation draw into (factor_name, scalar) pairs.

    A numeric vector becomes one pair per component, each named key[i]. A bare number or string
    becomes a single pair under key. A bool is treated as a categorical label rather than a 0/1
    number.

    Args:
        key: The variation name, asset.variation.
        value: The recorded draw for one episode.
        row_index: Source row index, used in error messages.
        jsonl_path: Source path, used in error messages.

    Returns:
        The (factor_name, scalar) pairs this draw contributes.
    """
    assert isinstance(value, (bool, int, float, str, list, tuple)), (
        f"Variation {key!r} in row {row_index} of {jsonl_path} has unsupported value type "
        f"{type(value).__name__}: {value!r}. Expected a number, string, or numeric vector."
    )
    # bool is an int subclass, so check it before int/float and keep it categorical.
    if isinstance(value, bool):
        return [(key, str(value))]
    if isinstance(value, (int, float)):
        return [(key, float(value))]
    if isinstance(value, str):
        return [(key, value)]
    # list / tuple → one continuous scalar factor per component.
    # TODO(cvolk): components are named with an opaque positional suffix (key[0], key[1], ...),
    # so plots can't tell e.g. a camera's lateral axis from its depth axis. Follow-up PR: have
    # the recorder emit semantic component names (e.g. camera ROS frame x_right/y_down/z_forward)
    # rather than a bare vector, so the labels flow through this generic reader unchanged.
    assert len(value) > 0, f"Variation {key!r} in row {row_index} of {jsonl_path} is an empty list."
    pairs: list[tuple[str, float | str]] = []
    for component_index, component in enumerate(value):
        assert isinstance(component, (int, float)) and not isinstance(component, bool), (
            f"Variation {key!r} in row {row_index} of {jsonl_path} is a vector with a non-numeric "
            f"component at index {component_index}: {component!r}. Vector variations must be all-numeric."
        )
        pairs.append((f"{key}[{component_index}]", float(component)))
    return pairs


def _build_dataset_from_episode_rows(
    rows: list[dict],
    outcome_names: list[str] | tuple[str, ...],
    jsonl_path: str | Path,
    factor_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[FactorSpec], torch.Tensor, torch.Tensor]:
    """Discover the factors from the rows and build the theta and x tensors.

    Each variation draw becomes a factor (see _flatten_variation_value), typed from its values:
    numeric is continuous, string is categorical. theta is continuous columns first then
    categorical (integer-coded), and x has one column per outcome. Every row must record the same
    factors. A factor that never varied is dropped, and an all-constant input raises.

    Args:
        rows: Parsed episode_results records, one per episode.
        outcome_names: Top-level field name(s) to read as outcomes.
        jsonl_path: Source path, used in error messages.
        factor_names: Which variations keys to keep, a vector keeping every component. None keeps all.

    Returns:
        The discovered factors, continuous-first, and the theta / x tensors.
    """
    selected = set(factor_names) if factor_names is not None else None
    if selected is not None:
        available = set(rows[0].get("variations", {}))
        missing = selected - available
        assert not missing, (
            f"Requested factor(s) {sorted(missing)} not found in {jsonl_path}; "
            f"available variations: {sorted(available)}."
        )

    factor_kinds: dict[str, str] = {}  # factor name → "continuous" | "categorical"
    factor_values: dict[str, list[float | str]] = {}  # factor name → per-row value, in row order
    factor_order: list[str] = []  # factor names in first-seen order, for a stable schema

    for row_index, row in enumerate(rows):
        assert "variations" in row, (
            f"Row {row_index} of {jsonl_path} has no 'variations' block; episode_results rows must "
            "carry recorded variation draws."
        )
        seen_in_row: set[str] = set()
        for key, value in row["variations"].items():
            if selected is not None and key not in selected:
                continue
            for factor_name, scalar in _flatten_variation_value(key, value, row_index, jsonl_path):
                kind = "categorical" if isinstance(scalar, str) else "continuous"
                if factor_name not in factor_kinds:
                    assert row_index == 0, (
                        f"Factor {factor_name!r} first appears in row {row_index} of {jsonl_path}; "
                        "every episode must record the same variations."
                    )
                    factor_kinds[factor_name] = kind
                    factor_values[factor_name] = []
                    factor_order.append(factor_name)
                assert factor_kinds[factor_name] == kind, (
                    f"Factor {factor_name!r} is {factor_kinds[factor_name]} in earlier rows but {kind} "
                    f"in row {row_index} of {jsonl_path}; a variation must keep a single type."
                )
                factor_values[factor_name].append(scalar)
                seen_in_row.add(factor_name)

        missing = [name for name in factor_order if name not in seen_in_row]
        assert not missing, (
            f"Row {row_index} of {jsonl_path} is missing factor(s) {sorted(missing)}; "
            "every episode must record the same variations."
        )
        for name in outcome_names:
            assert name in row, (
                f"Row {row_index} of {jsonl_path} is missing outcome field {name!r} "
                f"(requested outcomes: {list(outcome_names)})."
            )

    assert factor_order, f"No factors discovered in {jsonl_path}: every row's 'variations' block was empty."

    # Continuous factors lead theta, then categorical.
    continuous_names = [name for name in factor_order if factor_kinds[name] == "continuous"]
    categorical_names = [name for name in factor_order if factor_kinds[name] == "categorical"]

    # Drop factors that took a single value: they carry no information, and a constant categorical
    # breaks the estimator fit.
    factors: list[FactorSpec] = []
    columns: list[torch.Tensor] = []
    dropped: list[str] = []
    for name in continuous_names:
        values = factor_values[name]
        lo, hi = min(values), max(values)
        if lo == hi:
            dropped.append(name)
            continue
        factors.append(FactorSpec(name=name, type=FactorType.CONTINUOUS, range=[(lo, hi)]))
        columns.append(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
    for name in categorical_names:
        choices = sorted(set(factor_values[name]))
        if len(choices) == 1:
            dropped.append(name)
            continue
        # The analysis assumes factors were drawn from the uniform prior. Uneven draw counts per
        # choice leak into the posterior (a no-effect factor then tracks its sampling frequency),
        # so warn when the most-sampled choice exceeds the least by 1.5x or more.
        counts: dict[str, int] = {}
        for value in factor_values[name]:
            counts[value] = counts.get(value, 0) + 1
        if max(counts.values()) >= 1.5 * min(counts.values()):
            ordered_counts = {choice: counts[choice] for choice in choices}
            print(
                f"[WARNING] Categorical factor {name!r} was sampled unevenly across its choices "
                f"({ordered_counts}). Its posterior reflects this sampling frequency, not only its effect "
                "on the outcome. Balance the draws per choice for an unbiased result."
            )
        code_of = {choice: code for code, choice in enumerate(choices)}
        factors.append(FactorSpec(name=name, type=FactorType.CATEGORICAL, choices=choices))
        columns.append(
            torch.tensor([code_of[value] for value in factor_values[name]], dtype=torch.float32).unsqueeze(1)
        )

    if dropped:
        print(
            f"[INFO] Dropped {len(dropped)} constant factor(s) (single value across all episodes): {sorted(dropped)}."
        )
    assert factors, (
        f"All discovered factors in {jsonl_path} are constant (each took a single value across all "
        "episodes); nothing to analyze. Vary at least one factor."
    )

    theta = torch.cat(columns, dim=1)
    x = torch.cat(
        [torch.tensor([float(row[name]) for row in rows], dtype=torch.float32).unsqueeze(1) for name in outcome_names],
        dim=1,
    )
    return factors, theta, x
