# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end sensitivity-analysis tests on synthetic data with a known ground truth.

Each test fits a SensitivityAnalyzer on a dataset whose factor→outcome relationships are
planted by the synthetic module (brighter / lighter / closer / cube / oak raise success),
then asserts the posterior conditioned on success recovers them. The data is built in
memory, so these run on CPU without Isaac Sim. They cover both estimator paths: MNPE for a
mixed schema, NPE for a continuous-only one (2-D theta).
"""

from __future__ import annotations

import json
import numpy as np
import torch

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.tests.sensitivity_synthetic import (
    CAMERA_DISTANCE,
    GRASP_OFFSET,
    LIGHT,
    MATERIAL,
    OBJECT_MASS,
    OBJECT_TYPE,
    make_continuous_dataset,
    make_mixed_dataset,
)

_NUM_SAMPLES = 5000


def _factor_samples(analyzer: SensitivityAnalyzer, samples: torch.Tensor, factor_name: str) -> np.ndarray:
    """Pull one factor's column out of a posterior-sample tensor as a 1-D numpy array."""
    return samples[:, analyzer.dataset.factor_columns[factor_name]].squeeze(-1).cpu().numpy()


def _midpoint(factor) -> float:
    """Midpoint of a continuous factor's range — the threshold a recovered mean should beat."""
    low, high = factor.value_range
    return 0.5 * (low + high)


def _most_likely_choice(analyzer, samples, factor_name: str, choices: list[str]) -> str:
    """The categorical choice the posterior favors (mode over rounded integer-coded samples)."""
    codes = np.clip(np.round(_factor_samples(analyzer, samples, factor_name)), 0, len(choices) - 1).astype(int)
    probabilities = np.bincount(codes, minlength=len(choices)) / len(codes)
    return choices[int(probabilities.argmax())]


def test_mnpe_recovers_all_planted_effects():
    """Mixed continuous + categorical (MNPE): recover every planted effect at once."""
    dataset = make_mixed_dataset(seed=0)
    analyzer = SensitivityAnalyzer(dataset)
    assert analyzer._select_inference_class().__name__ == "MNPE", "mixed schema should select MNPE"

    torch.manual_seed(0)
    analyzer.fit()
    samples = analyzer.sample_posterior(num_samples=_NUM_SAMPLES)  # conditions on success=1 by default

    # Continuous effects: brighter light, a lighter object, and a closer camera raise success.
    assert _factor_samples(analyzer, samples, "light_intensity").mean() > _midpoint(LIGHT)
    assert _factor_samples(analyzer, samples, "object_mass").mean() < _midpoint(OBJECT_MASS)
    assert _factor_samples(analyzer, samples, "camera_distance").mean() < _midpoint(CAMERA_DISTANCE)

    # Categorical effects: cube and oak are the planted best choices.
    assert _most_likely_choice(analyzer, samples, "object_type", OBJECT_TYPE.choices) == "cube"
    assert _most_likely_choice(analyzer, samples, "table_material", MATERIAL.choices) == "oak"


def test_npe_recovers_two_continuous_effects():
    """Two continuous factors (NPE): recover that bright light and a small grasp offset drive success."""
    dataset = make_continuous_dataset(seed=0)
    analyzer = SensitivityAnalyzer(dataset)
    assert analyzer._select_inference_class().__name__.startswith("NPE"), "continuous-only schema should select NPE"

    torch.manual_seed(0)
    analyzer.fit()
    samples = analyzer.sample_posterior(num_samples=_NUM_SAMPLES)  # conditions on success=1 by default

    # Brighter light raises success → light posterior skews high.
    assert _factor_samples(analyzer, samples, "light_intensity").mean() > _midpoint(LIGHT)
    # A smaller grasp offset raises success → offset posterior skews low.
    assert _factor_samples(analyzer, samples, "grasp_offset").mean() < _midpoint(GRASP_OFFSET)


def _write_jsonl(path, rows: list[dict]) -> None:
    """Write one JSON object per line to ``path``."""
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_from_files_parses_mixed_schema_and_builds_tensors(tmp_path):
    """from_files parses a factors.yaml + episode_summary.jsonl into the expected theta / x layout."""
    factors_yaml = tmp_path / "factors.yaml"
    factors_yaml.write_text(
        "factors:\n"
        "  light_intensity:\n"
        "    type: continuous\n"
        "    range: [[0.0, 1000.0]]\n"
        "  pick_up_object:\n"
        "    type: categorical\n"
        "    choices: [cube, can]\n",
        encoding="utf-8",
    )
    jsonl = tmp_path / "episode_summary.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"arena_env_args": {"light_intensity": 250.0, "pick_up_object": "cube"}, "outcomes": {"success": 1}},
            {"arena_env_args": {"light_intensity": 750.0, "pick_up_object": "can"}, "outcomes": {"success": 0}},
            {"arena_env_args": {"light_intensity": 500.0, "pick_up_object": "cube"}, "outcomes": {"success": 1}},
        ],
    )

    dataset = SensitivityDataset.from_files(factors_yaml, jsonl, outcome_names=["success"])

    # Schema parsed with the declared structure.
    factors_by_name = {factor.name: factor for factor in dataset.schema.factors}
    assert factors_by_name["light_intensity"].type == "continuous"
    assert factors_by_name["light_intensity"].range == [(0.0, 1000.0)]
    assert factors_by_name["pick_up_object"].type == "categorical"
    assert factors_by_name["pick_up_object"].choices == ["cube", "can"]

    # Continuous-first theta layout; categorical integer-coded by its index into choices.
    assert dataset.theta.shape == (3, 2)
    assert dataset.x.shape == (3, 1)
    assert dataset.factor_columns == {"light_intensity": slice(0, 1), "pick_up_object": slice(1, 2)}
    assert dataset.theta[:, 0].tolist() == [250.0, 750.0, 500.0]
    assert dataset.theta[:, 1].tolist() == [0.0, 1.0, 0.0]  # cube -> 0, can -> 1
    assert dataset.x[:, 0].tolist() == [1.0, 0.0, 1.0]


def test_from_files_infers_missing_continuous_range(tmp_path):
    """A continuous factor with no declared range gets [min, max] inferred from the observed values."""
    factors_yaml = tmp_path / "factors.yaml"
    factors_yaml.write_text("factors:\n  light_intensity:\n    type: continuous\n", encoding="utf-8")
    jsonl = tmp_path / "episode_summary.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"arena_env_args": {"light_intensity": 30.0}, "outcomes": {"success": 0}},
            {"arena_env_args": {"light_intensity": 90.0}, "outcomes": {"success": 1}},
        ],
    )

    dataset = SensitivityDataset.from_files(factors_yaml, jsonl, outcome_names=["success"])

    assert dataset.schema.factors[0].range == [(30.0, 90.0)]
