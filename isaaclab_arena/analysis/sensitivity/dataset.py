# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import torch
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class FactorSpec:
    """One factor's schema as declared in ``factors.yaml``.

    Continuous factors carry a ``range`` (one ``[low, high]`` pair per dim); categorical
    factors carry ``choices`` (a list of string labels, integer-encoded by index in theta).
    """

    name: str
    type: Literal["continuous", "categorical"]
    dim: int = 1
    range: list[list[float]] | None = None  # one [low, high] pair per dim, continuous only
    choices: list[str] | None = None  # categorical only


@dataclass
class OutcomeSpec:
    """One outcome's schema (just a name and a type hint; the loader treats all as float)."""

    name: str
    type: str  # "bool", "float", "int" — informational; loader treats all as float


@dataclass
class SliceSpec:
    """Where a dataset came from: which policy ran which task on which embodiment.

    Read from the ``slice:`` block of ``factors.yaml`` and shown in the report title.
    It just labels the data — nothing checks it against the rows.
    """

    policy: str
    task: str
    embodiment: str


@dataclass
class FactorSchema:
    """Parsed ``factors.yaml`` — slice + factor list + outcome list."""

    slice: SliceSpec
    factors: list[FactorSpec]
    outcomes: list[OutcomeSpec]

    @classmethod
    def from_yaml(cls, path: str | Path) -> FactorSchema:
        """Load a ``factors.yaml`` from disk into a typed ``FactorSchema``.

        The YAML must have three top-level blocks: ``slice`` (policy/task/embodiment),
        ``factors`` (one entry per varied input), and ``outcomes`` (one entry per
        measured output). Each factor's ``type`` must be ``continuous`` or ``categorical``.
        """
        with open(path, encoding="utf-8") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        assert isinstance(yaml_data, dict), f"factors.yaml at {path} must be a mapping at top level"
        for required_key in ("slice", "factors", "outcomes"):
            assert required_key in yaml_data, f"factors.yaml at {path} is missing top-level `{required_key}:` block"

        slice_block = yaml_data["slice"]
        for required_key in ("policy", "task", "embodiment"):
            assert (
                required_key in slice_block
            ), f"factors.yaml at {path} `slice:` block is missing `{required_key}` (need policy/task/embodiment)"
        slice_spec = SliceSpec(
            policy=slice_block["policy"],
            task=slice_block["task"],
            embodiment=slice_block["embodiment"],
        )

        factors: list[FactorSpec] = []
        for factor_name, factor_block in yaml_data["factors"].items():
            assert "type" in factor_block, (
                f"factors.yaml at {path} factor {factor_name!r} is missing required `type:` field"
                " (expected 'continuous' or 'categorical')"
            )
            factor_type = factor_block["type"]
            assert factor_type in ("continuous", "categorical"), (
                f"factors.yaml at {path} factor {factor_name!r} has unknown type {factor_type!r};"
                " expected 'continuous' or 'categorical'"
            )
            factors.append(
                FactorSpec(
                    name=factor_name,
                    type=factor_type,
                    dim=factor_block.get("dim", 1),
                    range=factor_block.get("range"),
                    choices=factor_block.get("choices"),
                )
            )

        outcomes = [
            OutcomeSpec(name=outcome_name, type=outcome_block.get("type", "float"))
            for outcome_name, outcome_block in yaml_data["outcomes"].items()
        ]

        return cls(slice=slice_spec, factors=factors, outcomes=outcomes)

    @property
    def total_factor_dim(self) -> int:
        """Total width of theta — sum of ``dim`` over continuous factors plus 1 per categorical."""
        return sum(factor.dim if factor.type == "continuous" else 1 for factor in self.factors)

    @property
    def factor_columns(self) -> dict[str, slice]:
        """Map factor name → its column slice in theta.

        Continuous factors occupy the leading columns (``dim`` each), then each categorical
        factor occupies one trailing column. This continuous-first layout is what sbi's
        mixed density estimator expects.
        """
        continuous_factors = [factor for factor in self.factors if factor.type == "continuous"]
        categorical_factors = [factor for factor in self.factors if factor.type == "categorical"]
        column_slices: dict[str, slice] = {}
        column_index = 0
        for factor in continuous_factors + categorical_factors:
            column_width = factor.dim if factor.type == "continuous" else 1
            column_slices[factor.name] = slice(column_index, column_index + column_width)
            column_index += column_width
        return column_slices


class SensitivityDataset:
    """A ``FactorSchema`` paired with its per-episode ``theta`` (factors) and ``x`` (outcomes).

    The object is a pure container: it holds the schema and the two tensors, and exposes
    the ``prior`` and column layouts an analyzer consumes. It can be built two ways:

      - :meth:`from_files` — parse a ``factors.yaml`` / ``episode_summary.jsonl`` pair
        (the path eval runs take).
      - the constructor — wrap in-memory tensors directly (what a synthetic simulator or
        a unit test takes). The tensors must already be in the layout :meth:`factor_columns`
        describes: continuous columns first, then one integer-coded column per categorical.
    """

    def __init__(self, schema: FactorSchema, theta: torch.Tensor, x: torch.Tensor):
        """Wrap an in-memory ``schema`` plus its ``theta`` / ``x`` tensors, validating shapes.

        Args:
            schema: The parsed factor/outcome schema. Continuous factors must carry a
                ``range``; categorical factors must carry ``choices``.
            theta: ``(num_episodes, total_factor_dim)`` factor matrix, continuous-first.
            x: ``(num_episodes, num_outcomes)`` outcome matrix.
        """
        assert theta.ndim == 2 and x.ndim == 2, f"theta and x must be 2D; got {theta.shape} and {x.shape}"
        assert theta.shape[0] == x.shape[0], f"theta/x row counts disagree: {theta.shape[0]} vs {x.shape[0]}"
        assert theta.shape[0] > 0, "Dataset is empty (no episodes)"
        assert (
            theta.shape[1] == schema.total_factor_dim
        ), f"theta has {theta.shape[1]} columns but schema declares {schema.total_factor_dim} factor dims"
        assert x.shape[1] == len(
            schema.outcomes
        ), f"x has {x.shape[1]} columns but schema declares {len(schema.outcomes)} outcomes"
        self.schema = schema
        self._theta = theta
        self._x = x

    @classmethod
    def from_files(cls, factors_yaml: str | Path, jsonl_path: str | Path) -> SensitivityDataset:
        """Build a dataset from a ``factors.yaml`` schema and an ``episode_summary.jsonl``.

        Parses and validates both, infers any missing continuous range from the data, and
        assembles the ``theta`` / ``x`` tensors in the layout the analyzers expect.
        """
        schema = FactorSchema.from_yaml(factors_yaml)

        jsonl_text = Path(jsonl_path).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
        assert len(rows) > 0, f"Empty episode_summary.jsonl at {jsonl_path}"

        _validate_rows(schema, rows, jsonl_path)
        _infer_missing_factor_ranges(schema, rows)

        theta = _build_factor_tensor(schema, rows)
        x = _build_outcome_tensor(schema, rows)
        return cls(schema, theta, x)

    @property
    def theta(self) -> torch.Tensor:
        """``(num_episodes, total_factor_dim)`` matrix of factor values, one row per episode.

        This is the "input" sbi infers a posterior over. Column layout is given by
        ``factor_columns`` — continuous factors first, then categoricals (integer-coded).
        """
        return self._theta

    @property
    def x(self) -> torch.Tensor:
        """``(num_episodes, num_outcomes)`` matrix of outcome values, one row per episode.

        This is what the analyzer conditions queries on. The analyzer typically selects a
        single outcome column at fit time (e.g. ``success_rate``) and asks
        "what theta values were consistent with observing this outcome?"
        """
        return self._x

    @property
    def num_episodes(self) -> int:
        """Number of episodes (rows) in the dataset."""
        return self._theta.shape[0]

    @property
    def factor_columns(self) -> dict[str, slice]:
        """Map factor name → its column slice in theta. Same as ``schema.factor_columns``."""
        return self.schema.factor_columns

    @property
    def outcome_columns(self) -> dict[str, int]:
        """Map outcome name → its column index in x."""
        return {outcome.name: index for index, outcome in enumerate(self.schema.outcomes)}

    @property
    def has_categorical_factors(self) -> bool:
        """True iff the schema declares at least one categorical factor."""
        return any(factor.type == "categorical" for factor in self.schema.factors)

    @property
    def prior(self):
        """Uniform prior (``sbi.utils.BoxUniform``) over all factor dims.

        Continuous factors use their ``[low, high]`` range; categorical factors use
        ``[0, num_choices - 1]`` over their integer codes, in continuous-first order.
        """
        # Import sbi lazily so loading the dataset does not pay the sbi import cost
        # unless an analyzer actually runs.
        from sbi.utils import BoxUniform

        low_bounds: list[float] = []
        high_bounds: list[float] = []

        # Continuous bounds (one [low, high] per dim).
        for factor in self.schema.factors:
            if factor.type != "continuous":
                continue
            assert factor.range is not None, f"Factor {factor.name!r} has no range and was not inferred"
            for dim_low, dim_high in factor.range:
                low_bounds.append(float(dim_low))
                high_bounds.append(float(dim_high))

        # Categorical factor bounds: [0, num_choices - 1] per factor (one column).
        for factor in self.schema.factors:
            if factor.type != "categorical":
                continue
            assert (
                factor.choices is not None and len(factor.choices) > 0
            ), f"Categorical factor {factor.name!r} has no `choices:` block"
            low_bounds.append(0.0)
            high_bounds.append(float(len(factor.choices) - 1))

        return BoxUniform(
            low=torch.tensor(low_bounds, dtype=torch.float32),
            high=torch.tensor(high_bounds, dtype=torch.float32),
        )


def _validate_rows(schema: FactorSchema, rows: list[dict], jsonl_path: str | Path) -> None:
    """Assert every JSONL row carries the schema's declared factor and outcome keys.

    The declared names need only be a subset of each row's ``arena_env_args`` / ``outcomes``;
    extra keys are ignored. Raises pointing at the first offending row.
    """
    expected_factor_names = {factor.name for factor in schema.factors}
    expected_outcome_names = {outcome.name for outcome in schema.outcomes}
    for row_index, row in enumerate(rows):
        assert (
            "arena_env_args" in row and "outcomes" in row
        ), f"Row {row_index} of {jsonl_path} missing arena_env_args/outcomes block"
        missing_factor_names = expected_factor_names - set(row["arena_env_args"].keys())
        assert not missing_factor_names, (
            f"Row {row_index} of {jsonl_path} is missing factor(s) "
            f"{sorted(missing_factor_names)} from its arena_env_args block; "
            f"factors.yaml declares: {sorted(expected_factor_names)}"
        )
        missing_outcome_names = expected_outcome_names - set(row["outcomes"].keys())
        assert (
            not missing_outcome_names
        ), f"Row {row_index} of {jsonl_path} missing outcomes {sorted(missing_outcome_names)}"


def _infer_missing_factor_ranges(schema: FactorSchema, rows: list[dict]) -> None:
    """Fill any continuous factor's missing ``range`` from the observed min/max.

    A ``range`` declared in factors.yaml takes precedence and is left untouched.
    """
    for factor in schema.factors:
        if factor.type != "continuous" or factor.range is not None:
            continue
        if factor.dim != 1:
            raise NotImplementedError(
                "Range inference for vector factors (dim > 1) is not implemented;"
                f" factor {factor.name!r} has dim={factor.dim}"
            )
        observed_values = [float(row["arena_env_args"][factor.name]) for row in rows]
        factor.range = [[min(observed_values), max(observed_values)]]


def _build_factor_tensor(schema: FactorSchema, rows: list[dict]) -> torch.Tensor:
    """Assemble the per-episode factor matrix ``theta``.

    Continuous columns first (one per dim), then one column per categorical factor with its
    value integer-coded as a ``float32`` index into ``FactorSpec.choices``.
    """
    continuous_factors = [factor for factor in schema.factors if factor.type == "continuous"]
    categorical_factors = [factor for factor in schema.factors if factor.type == "categorical"]

    factor_columns: list[torch.Tensor] = []

    # Continuous columns come first (sbi MNPE convention).
    for factor in continuous_factors:
        if factor.dim != 1:
            raise NotImplementedError(
                "Vector continuous factors (dim > 1) are not yet supported;"
                f" factor {factor.name!r} has dim={factor.dim}"
            )
        raw_values = [float(row["arena_env_args"][factor.name]) for row in rows]
        factor_column = torch.tensor(raw_values, dtype=torch.float32).unsqueeze(1)
        factor_columns.append(factor_column)

    # Categorical columns: integer-code each string value as its index in FactorSpec.choices.
    for factor in categorical_factors:
        assert (
            factor.choices is not None and len(factor.choices) > 0
        ), f"Categorical factor {factor.name!r} has no `choices:` block in factors.yaml"
        choice_to_code = {choice: code for code, choice in enumerate(factor.choices)}
        category_codes: list[int] = []
        for row_index, row in enumerate(rows):
            value = row["arena_env_args"][factor.name]
            assert (
                value in choice_to_code
            ), f"Row {row_index} factor {factor.name!r} has value {value!r} not in declared choices {factor.choices}"
            category_codes.append(choice_to_code[value])
        factor_column = torch.tensor(category_codes, dtype=torch.float32).unsqueeze(1)
        factor_columns.append(factor_column)

    if factor_columns:
        return torch.cat(factor_columns, dim=1)
    return torch.zeros((len(rows), 0), dtype=torch.float32)


def _build_outcome_tensor(schema: FactorSchema, rows: list[dict]) -> torch.Tensor:
    """Assemble the per-episode outcome matrix ``x`` (one column per declared outcome).

    Each outcome value is cast to float; bool outcomes become 0.0/1.0. The analyzer usually
    selects a single outcome column at fit time and conditions queries on it.
    """
    outcome_column_tensors = [
        torch.tensor([float(row["outcomes"][outcome.name]) for row in rows], dtype=torch.float32).unsqueeze(1)
        for outcome in schema.outcomes
    ]
    return torch.cat(outcome_column_tensors, dim=1)
