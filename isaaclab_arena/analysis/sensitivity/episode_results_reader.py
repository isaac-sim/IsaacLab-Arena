# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read an episode_results.jsonl (the per-episode recorder's output) into a SensitivityDataset.

This module is the only place that knows the recorder's on-disk format, so dataset.py stays a
pure in-memory container.
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from typing import Any

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, FactorType, SensitivityDataset

_IMBALANCE_WARN_RATIO = 1.5
"""Warn when a categorical's most-sampled choice exceeds its least-sampled one by at least this factor."""


def dataset_from_episode_results(
    jsonl_path: str | Path,
    outcome_names: list[str] | tuple[str, ...] = ("success",),
    factor_names: list[str] | tuple[str, ...] | None = None,
) -> SensitivityDataset:
    """Build a SensitivityDataset from an episode_results.jsonl, discovering the factors from the data.

    Each line is one episode. The variations block holds the sampled factor draws, and the
    top-level fields named by outcome_names hold the outcomes. Other top-level fields are ignored.
    A number becomes a continuous factor, a numeric vector becomes one continuous factor per
    component (named key[i]), and a string becomes a categorical factor over its observed labels.

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
    rows = _read_rows(jsonl_path)
    factor_kinds, factor_values, factor_order = _discover_factor_values(rows, outcome_names, jsonl_path, factor_names)
    factors, theta = _build_factor_columns(factor_kinds, factor_values, factor_order, jsonl_path)
    x = _build_outcome_columns(rows, outcome_names, jsonl_path)
    return SensitivityDataset(factors, theta, x, outcome_names)


def _read_rows(jsonl_path: str | Path) -> list[dict]:
    """Parse the JSONL file into a non-empty list of episode records."""
    jsonl_text = Path(jsonl_path).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
    assert len(rows) > 0, f"Empty episode_results.jsonl at {jsonl_path}"
    return rows


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


def _discover_factor_values(
    rows: list[dict],
    outcome_names: list[str] | tuple[str, ...],
    jsonl_path: str | Path,
    factor_names: list[str] | tuple[str, ...] | None,
) -> tuple[dict[str, str], dict[str, list[float | str]], list[str]]:
    """Scan the rows into per-factor value lists, checking the recorder contract.

    Flattens each row's variation draws (see _flatten_variation_value), keeps only the requested
    factor_names if given, and asserts every episode records the same factors and the requested
    outcomes. Returns the factor kinds, the per-row values, and the first-seen order.
    """
    selected = set(factor_names) if factor_names is not None else None
    if selected is not None:
        first_variations = rows[0].get("variations")
        assert isinstance(
            first_variations, dict
        ), f"Row 0 of {jsonl_path} has no 'variations' block (or it is not a JSON object)."
        available = set(first_variations)
        missing = selected - available
        assert not missing, (
            f"Requested factor(s) {sorted(missing)} not found in {jsonl_path}; "
            f"available variations: {sorted(available)}."
        )

    factor_kinds: dict[str, str] = {}  # factor name → "continuous" | "categorical"
    factor_values: dict[str, list[float | str]] = {}  # factor name → per-row value, in row order
    factor_order: list[str] = []  # factor names in first-seen order, for a stable schema

    for row_index, row in enumerate(rows):
        assert "variations" in row and isinstance(row["variations"], dict), (
            f"Row {row_index} of {jsonl_path} has no 'variations' block (or it is not a JSON object); "
            "episode_results rows must carry recorded variation draws."
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

        missing_in_row = [name for name in factor_order if name not in seen_in_row]
        assert not missing_in_row, (
            f"Row {row_index} of {jsonl_path} is missing factor(s) {sorted(missing_in_row)}; "
            "every episode must record the same variations."
        )
        for name in outcome_names:
            assert name in row, (
                f"Row {row_index} of {jsonl_path} is missing outcome field {name!r} "
                f"(requested outcomes: {list(outcome_names)})."
            )

    assert factor_order, f"No factors discovered in {jsonl_path}: every row's 'variations' block was empty."
    return factor_kinds, factor_values, factor_order


def _build_factor_columns(
    factor_kinds: dict[str, str],
    factor_values: dict[str, list[float | str]],
    factor_order: list[str],
    jsonl_path: str | Path,
) -> tuple[list[FactorSpec], torch.Tensor]:
    """Turn the discovered per-factor values into the factor specs and the theta matrix.

    Continuous factors lead theta, then categoricals (integer-coded). A factor that took a single
    value is dropped (it carries no information, and a constant categorical breaks the estimator
    fit), and an all-constant input raises.
    """
    continuous_names = [name for name in factor_order if factor_kinds[name] == "continuous"]
    categorical_names = [name for name in factor_order if factor_kinds[name] == "categorical"]

    factors: list[FactorSpec] = []
    columns: list[torch.Tensor] = []
    dropped: list[str] = []
    for name in continuous_names:
        values = factor_values[name]
        lo, hi = min(values), max(values)
        if lo == hi:
            dropped.append(name)
            continue
        factors.append(FactorSpec(name=name, type=FactorType.CONTINUOUS, range=(lo, hi)))
        columns.append(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
    for name in categorical_names:
        choices = sorted(set(factor_values[name]))
        if len(choices) == 1:
            dropped.append(name)
            continue
        _warn_if_unevenly_sampled(name, factor_values[name], choices)
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
        "episodes). Nothing to analyze. Vary at least one factor."
    )
    return factors, torch.cat(columns, dim=1)


def _warn_if_unevenly_sampled(name: str, values: list[float | str], choices: list[str]) -> None:
    """Warn when a categorical's choices were sampled unevenly, since that biases its posterior.

    The analysis assumes factors were drawn from the uniform prior. Uneven draw counts per choice
    leak into the posterior (a no-effect factor then tracks its sampling frequency), so warn once
    the imbalance reaches _IMBALANCE_WARN_RATIO.
    """
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if max(counts.values()) >= _IMBALANCE_WARN_RATIO * min(counts.values()):
        ordered_counts = {choice: counts[choice] for choice in choices}
        print(
            f"[WARNING] Categorical factor {name!r} was sampled unevenly across its choices "
            f"({ordered_counts}). Its posterior reflects this sampling frequency, not only its effect "
            "on the outcome. Balance the draws per choice for an unbiased result."
        )


def _build_outcome_columns(
    rows: list[dict], outcome_names: list[str] | tuple[str, ...], jsonl_path: str | Path
) -> torch.Tensor:
    """Stack the requested top-level outcome fields into the x matrix, one column per outcome.

    Asserts each outcome value is numeric or boolean, so a stray non-numeric outcome fails with
    the same row-and-path context as a bad variation rather than a bare cast error.
    """
    columns: list[torch.Tensor] = []
    for name in outcome_names:
        values: list[float] = []
        for row_index, row in enumerate(rows):
            value = row[name]
            assert isinstance(value, (bool, int, float)), (
                f"Outcome {name!r} in row {row_index} of {jsonl_path} is {type(value).__name__} {value!r}; "
                "outcomes must be numeric or boolean."
            )
            values.append(float(value))
        columns.append(torch.tensor(values, dtype=torch.float32).unsqueeze(1))
    return torch.cat(columns, dim=1)
