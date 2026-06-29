# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass
from enum import Enum


class FactorType(str, Enum):
    """Whether a factor's values are continuous (numeric range) or categorical (labelled choices)."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class FactorSpec:
    """One varied input — a lighting level, a camera-offset axis, a background choice, and so on.

    Each factor occupies one column of the dataset's factor matrix theta (see SensitivityDataset).
    A continuous factor carries a range, the (low, high) it was swept over. A categorical factor
    carries choices, the string labels it took, integer-encoded by their index in that column.
    """

    name: str
    type: FactorType
    range: tuple[float, float] | None = None  # (low, high), continuous only
    choices: list[str] | None = None  # categorical only

    def __post_init__(self) -> None:
        # Accept the raw string form (from YAML / callers) and normalize to the enum.
        self.type = FactorType(self.type)
        # JSON/YAML deliver the range as a list; normalize it to a tuple.
        if self.range is not None:
            self.range = tuple(self.range)


class SensitivityDataset:
    """The varied factors paired with their per-episode values (theta) and outcomes (x).

    theta is the factor matrix: one row per episode, one column per factor — continuous factors
    in the leading columns, then one integer-coded column per categorical factor. x is the
    matching outcome matrix, one row per episode and one column per outcome. The object is a pure
    in-memory container (the factor list plus the two tensors) and exposes the column layout an
    analyzer reads.
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
