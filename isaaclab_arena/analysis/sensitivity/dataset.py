# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
    """The ``(policy, task, embodiment)`` tuple a dataset comes from.

    MNPE/NPE assume a single data-generating source per analysis, so all rows in a
    dataset must belong to the same slice — enforced by the loader.
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
        """Map factor name → column slice in theta.

        Continuous factors occupy the leading columns (their ``dim`` columns each), then
        each categorical factor occupies one trailing column. This continuous-first
        ordering matches sbi's MNPE convention so the same theta layout works for both
        NPE (all-continuous) and MNPE (mixed).
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
    """Combines a ``factors.yaml`` schema with an ``episode_summary.jsonl`` data file.

    On construction:
      1. Parses the schema (factors + outcomes + slice metadata).
      2. Loads the JSONL rows (one row per episode).
      3. Validates that every row contains all declared factor and outcome keys.
      4. Fills any missing continuous ranges by inferring from observed min/max so the
         analyzer can always trust ``schema.factors[i].range`` to be populated.
      5. Builds the ``theta`` and ``x`` tensors that sbi (or the empirical analyzer)
         will consume.

    The four public attributes used by the analyzer (``theta``, ``x``, ``prior``,
    ``factor_columns``) are properties — recomputed lazily where appropriate.
    """

    def __init__(self, factors_yaml: str | Path, jsonl_path: str | Path):
        self.schema = FactorSchema.from_yaml(factors_yaml)

        jsonl_text = Path(jsonl_path).read_text(encoding="utf-8")
        self.rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
        assert len(self.rows) > 0, f"Empty episode_summary.jsonl at {jsonl_path}"

        self._validate_rows(jsonl_path)
        self._infer_missing_factor_ranges()

        self._theta = self._build_factor_tensor()
        self._x = self._build_outcome_tensor()

    def _validate_rows(self, jsonl_path: str | Path) -> None:
        """Assert every JSONL row carries the keys declared in the schema.

        The writer logs the *entire* arena_env_args dict per row, so the loader only
        requires that the schema's declared factor names are a *subset* of what's in
        ``row["arena_env_args"]`` — extra keys (other arena_env_args we don't analyze)
        are fine and ignored. Same superset-not-equality check for outcomes.

        Catches the most common authoring mistake: a factor declared in factors.yaml
        that the eval didn't actually vary or log. Surfaces a clear error pointing at
        the first offending row.
        """
        expected_factor_names = {factor.name for factor in self.schema.factors}
        expected_outcome_names = {outcome.name for outcome in self.schema.outcomes}
        for row_index, row in enumerate(self.rows):
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

    def _infer_missing_factor_ranges(self) -> None:
        """For any continuous factor without a declared ``range``, fill it from observed data.

        The prior bounds default to ``[min(values), max(values)]`` over the JSONL. Users
        who want a principled prior (e.g. matching the variation system's declared
        ``Uniform(low, high)``) should hand-author ``range`` in factors.yaml; that value
        takes precedence and this method skips them.
        """
        for factor in self.schema.factors:
            if factor.type != "continuous" or factor.range is not None:
                continue
            if factor.dim != 1:
                raise NotImplementedError(
                    "Range inference for vector factors (dim > 1) is not implemented;"
                    f" factor {factor.name!r} has dim={factor.dim}"
                )
            observed_values = [float(row["arena_env_args"][factor.name]) for row in self.rows]
            factor.range = [[min(observed_values), max(observed_values)]]

    def _build_factor_tensor(self) -> torch.Tensor:
        """Assemble the per-episode factor matrix ``theta``.

        Layout: continuous factors fill the leading columns (one column per dim), then
        each categorical factor fills one trailing column. Categorical values are
        encoded as ``float32`` integers ``0..num_choices-1`` per the index in
        ``FactorSpec.choices`` — sbi's MNPE expects exactly this layout (continuous-first,
        discrete columns as floats, the density estimator handles them as discrete).
        """
        continuous_factors = [factor for factor in self.schema.factors if factor.type == "continuous"]
        categorical_factors = [factor for factor in self.schema.factors if factor.type == "categorical"]

        factor_columns: list[torch.Tensor] = []

        # Continuous columns come first (sbi MNPE convention).
        for factor in continuous_factors:
            if factor.dim != 1:
                raise NotImplementedError(
                    "Vector continuous factors (dim > 1) are not yet supported;"
                    f" factor {factor.name!r} has dim={factor.dim}"
                )
            raw_values = [float(row["arena_env_args"][factor.name]) for row in self.rows]
            factor_column = torch.tensor(raw_values, dtype=torch.float32).unsqueeze(1)
            factor_columns.append(factor_column)

        # Categorical columns: integer-code each string value as its index in FactorSpec.choices.
        for factor in categorical_factors:
            assert (
                factor.choices is not None and len(factor.choices) > 0
            ), f"Categorical factor {factor.name!r} has no `choices:` block in factors.yaml"
            choice_to_code = {choice: code for code, choice in enumerate(factor.choices)}
            category_codes: list[int] = []
            for row_index, row in enumerate(self.rows):
                value = row["arena_env_args"][factor.name]
                assert value in choice_to_code, (
                    f"Row {row_index} factor {factor.name!r} has value {value!r}"
                    f" not in declared choices {factor.choices}"
                )
                category_codes.append(choice_to_code[value])
            factor_column = torch.tensor(category_codes, dtype=torch.float32).unsqueeze(1)
            factor_columns.append(factor_column)

        if factor_columns:
            return torch.cat(factor_columns, dim=1)
        return torch.zeros((len(self.rows), 0), dtype=torch.float32)

    def _build_outcome_tensor(self) -> torch.Tensor:
        """Assemble the per-episode outcome matrix ``x`` (one column per declared outcome).

        Each outcome value is cast to float; bool outcomes become 0.0/1.0. The analyzer
        usually selects a single outcome column at fit time and conditions queries on it.
        """
        outcome_column_tensors = [
            torch.tensor([float(row["outcomes"][outcome.name]) for row in self.rows], dtype=torch.float32).unsqueeze(1)
            for outcome in self.schema.outcomes
        ]
        return torch.cat(outcome_column_tensors, dim=1)

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
        """The uniform prior over all factor dims that the analyzer assumes.

        Built as a single ``sbi.utils.BoxUniform`` over the concatenated bounds in
        continuous-first / categorical-after order:
          - Continuous factor → uses the declared (or inferred) ``[low, high]`` per dim.
          - Categorical factor → uses ``[0, num_choices - 1]`` (the integer codes from
            ``_build_factor_tensor``); sbi MNPE's mixed density estimator treats them as
            discrete from there.

        sbi is imported lazily so loading the dataset doesn't pay the sbi import cost
        unless the analyzer actually runs.
        """
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
