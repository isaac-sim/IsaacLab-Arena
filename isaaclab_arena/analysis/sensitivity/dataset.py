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
    """One factor's schema. It occupies exactly one column of theta.

    Continuous factors carry a range (a single [low, high] pair); categorical
    factors carry choices (a list of string labels, integer-encoded by index in theta).
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

    The object is a pure container: it holds the factor list and the two tensors, and exposes
    the column layout an analyzer consumes. It can be built two ways:

      - from_episode_results — parse an episode_results.jsonl, discovering the factors from the
        recorded variation draws (the path eval runs take).
      - the constructor — wrap in-memory tensors directly (what a synthetic simulator or
        a unit test takes). The tensors must already be in the layout factor_columns
        describes: continuous columns first, then one integer-coded column per categorical.
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
            factors: The varied factors, one per theta column. Continuous factors must carry
                a range; categorical factors must carry choices.
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
    ) -> SensitivityDataset:
        """Build a dataset from an episode_results.jsonl, discovering the factor schema from the data.

        Each line is one episode. The ``variations`` block holds the sampled factor draws, and the
        top-level fields named by ``outcome_names`` hold the outcomes; any other top-level fields are
        ignored. Factor types are inferred from the values: a number is a continuous factor, a numeric
        vector becomes one continuous factor per component (named ``key[i]``), and a string is a
        categorical factor over its observed labels.

        Example line (one vector and one string factor, conditioned on ``success``):

            {"success": true,
             "variations": {"wrist_camera": [0.01, -0.02, 0.0], "hdr_image": "sunset"}}

        Args:
            jsonl_path: Path to the episode_results.jsonl (one JSON object per line).
            outcome_names: Which top-level field(s) per line to use as outcomes.

        Returns:
            A SensitivityDataset with theta / x in the continuous-first layout the analyzers expect.
        """
        jsonl_text = Path(jsonl_path).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
        assert len(rows) > 0, f"Empty episode_results.jsonl at {jsonl_path}"

        factors, theta, x = _build_dataset_from_episode_rows(rows, outcome_names, jsonl_path)
        return cls(factors, theta, x, outcome_names)

    @property
    def theta(self) -> torch.Tensor:
        """(num_episodes, num_factors) matrix of factor values, one row per episode.

        This is the "input" sbi infers a posterior over. Column layout is given by
        factor_columns — continuous factors first, then categoricals (integer-coded).
        """
        return self._theta

    @property
    def x(self) -> torch.Tensor:
        """(num_episodes, num_outcomes) matrix of outcome values, one row per episode.

        This is what the analyzer conditions queries on — "what factor values were consistent
        with observing these outcomes?". Columns are named by ``outcome_names``.
        """
        return self._x

    @property
    def num_episodes(self) -> int:
        """Number of episodes (rows) in the dataset."""
        return self._theta.shape[0]

    @property
    def factor_columns(self) -> dict[str, slice]:
        """Map factor name → its single-column slice in theta.

        Continuous factors take the leading columns, then categoricals — the continuous-first
        layout sbi's mixed density estimator expects. Each factor occupies exactly one column.
        """
        continuous = [factor for factor in self.factors if factor.type == "continuous"]
        categorical = [factor for factor in self.factors if factor.type == "categorical"]
        return {factor.name: slice(index, index + 1) for index, factor in enumerate(continuous + categorical)}

    def default_observation(self) -> torch.Tensor:
        """The default outcome vector to condition a query on: success (1) for every outcome.

        Outcomes are binary (0/1) in the current scope, so the natural default query is
        "what produced success?". Asserts the outcomes are binary, so adding a continuous
        outcome later fails loudly here instead of silently conditioning on a meaningless value.
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

    A numeric vector is split into one pair per component (name suffixed ``[i]``); a bare number
    or string yields a single pair under ``key``. Bools are treated as categorical labels, not
    0/1 continuous values.

    Args:
        key: The variation key (``"<asset>.<variation>"``).
        value: The recorded draw for one episode.
        row_index: Source row index, for error messages.
        jsonl_path: Source path, for error messages.

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
    rows: list[dict], outcome_names: list[str] | tuple[str, ...], jsonl_path: str | Path
) -> tuple[list[FactorSpec], torch.Tensor, torch.Tensor]:
    """Discover the factors and assemble theta / x directly from episode_results rows.

    Each row's ``variations`` entries are flattened into (factor_name, value) pairs (see
    _flatten_variation_value) and a factor's type is inferred from its values: numeric →
    continuous (range = observed [min, max]), string → categorical (choices = sorted observed
    labels). theta is laid out continuous-first with categoricals integer-coded; x has one column
    per outcome name. The factor set must be identical across every row. Factors that took a single
    value across all episodes are dropped (they carry no information); if none vary, this raises.

    Args:
        rows: Parsed episode_results records, one per episode.
        outcome_names: Top-level field name(s) to read as outcomes.
        jsonl_path: Source path, for error messages.

    Returns:
        The discovered factors (continuous-first) and the theta / x tensors.
    """
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

    # Continuous factors lead theta (sbi's mixed-estimator convention); categoricals follow.
    continuous_names = [name for name in factor_order if factor_kinds[name] == "continuous"]
    categorical_names = [name for name in factor_order if factor_kinds[name] == "categorical"]

    # Drop factors that took a single value across all episodes: they carry no information, and a
    # constant categorical would crash MNPE's mixed-density transform during fit.
    factors: list[FactorSpec] = []
    columns: list[torch.Tensor] = []
    dropped: list[str] = []
    for name in continuous_names:
        values = factor_values[name]
        if min(values) == max(values):
            dropped.append(name)
            continue
        factors.append(FactorSpec(name=name, type=FactorType.CONTINUOUS, range=[(min(values), max(values))]))
        columns.append(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
    for name in categorical_names:
        choices = sorted(set(factor_values[name]))
        if len(choices) == 1:
            dropped.append(name)
            continue
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
