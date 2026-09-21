# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify Isaac-Cap placement policies are declared in graph YAML."""

from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_checks_for_placer_params
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

_CAP_ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.isaac_cap


def _build_placer_params(data: dict):
    spec = ArenaEnvGraphSpec.model_construct(placer_params=data.get("placer_params"))
    return build_checks_for_placer_params(spec)


@pytest.mark.parametrize(
    ("yaml_name", "layout_family"),
    (("gear_easy.yaml", False), ("gear_easy_pair.yaml", True), ("gear_medium_train.yaml", True)),
)
def test_gear_mesh_placer_params(yaml_name: str, layout_family: bool):
    data = load_env_graph_spec_dict(_CAP_ROOT / "gear_insertion_v2" / yaml_name)

    assert data["placer_params"] == {
        "allow_best_loss_fallbacks": False,
        **({"min_unique_layouts_per_env": 1, "required_checks": []} if layout_family else {}),
        "solver_params": {"clearance_m": 0.0},
    }
    if layout_family:
        assert data["placer_params"]["required_checks"] == []
    else:
        assert "required_checks" not in data["placer_params"]
    params = _build_placer_params(data)
    assert not params.allow_best_loss_fallbacks
    assert params.solver_params.clearance_m == 0.0
    assert params.min_unique_layouts_per_env == (1 if layout_family else 5)
    assert params.required_checks == (set() if layout_family else None)


@pytest.mark.parametrize(
    ("yaml_name", "expected"),
    (
        ("syringe_both.yaml", {"allow_best_loss_fallbacks": False}),
        (
            "syringe_cluttered.yaml",
            {
                "allow_best_loss_fallbacks": False,
                "max_placement_attempts": 30,
                "solver_params": {"clearance_m": 0.015},
            },
        ),
    ),
)
def test_syringe_placer_params(yaml_name: str, expected: dict):
    data = load_env_graph_spec_dict(_CAP_ROOT / "syringe_sort" / "environments" / yaml_name)

    assert data["placer_params"] == expected
    params = _build_placer_params(data)
    assert not params.allow_best_loss_fallbacks
    if yaml_name == "syringe_cluttered.yaml":
        assert params.max_placement_attempts == 30
        assert params.solver_params.clearance_m == pytest.approx(0.015)
