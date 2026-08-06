# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test reading typed YAML Experiments before their Runs are composed."""

from pathlib import Path

import pytest

from isaaclab_arena.hydra.typed_experiment_loader import load_arena_experiment_from_yaml
from isaaclab_arena.hydra.typed_experiment_yaml_search import typed_experiment_requires_cameras
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

CAMERA_SENSITIVITY_EXPERIMENT_PATH = (
    Path(TestConstants.arena_environments_dir)
    / "experiment_configs"
    / "droid_pnp_camera_sensitivity_openpi_experiment.yaml"
)

_CAMERA_EXPERIMENT_YAML = """
runs:
  first:
    environment:
      type: pick_and_place_maple_table
      enable_cameras: {first_enable_cameras}
    policy:
      type: zero_action
  second:
    environment:
      type: pick_and_place_maple_table
      enable_cameras: {second_enable_cameras}
    policy:
      type: zero_action
"""


def _load_experiment(config_path: str | Path, overrides: list[str] | None = None):
    return load_arena_experiment_from_yaml(
        config_path,
        environment_cfg_types={"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
        policy_cfg_type_resolver=lambda name: {"zero_action": ZeroActionPolicyCfg}[name],
        overrides=overrides,
    )


def _write_experiment(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_experiment_without_declared_cameras_does_not_require_cameras(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  maple_table:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""",
    )

    assert typed_experiment_requires_cameras(config_path) is False


def test_shipped_camera_experiment_requires_cameras():
    assert typed_experiment_requires_cameras(CAMERA_SENSITIVITY_EXPERIMENT_PATH) is True


@pytest.mark.parametrize(
    ("first_enable_cameras", "second_enable_cameras", "overrides"),
    [
        (False, False, []),
        (True, False, []),
        (False, True, []),
        (True, True, []),
        (False, False, ["runs.second.environment.enable_cameras=true"]),
        (True, False, ["runs.first.environment.enable_cameras=false"]),
        (True, True, ["runs.first.environment.enable_cameras=false"]),
        (False, False, ["++runs.second.environment.enable_cameras=true"]),
        (True, False, ["runs.first.environment.light_intensity=750.0"]),
    ],
)
def test_camera_requirement_read_before_startup_matches_composed_experiment(
    tmp_path, first_enable_cameras, second_enable_cameras, overrides
):
    """The pre-startup read must agree with composition; the runner asserts on this invariant."""
    config_path = _write_experiment(
        tmp_path,
        _CAMERA_EXPERIMENT_YAML.format(
            first_enable_cameras=str(first_enable_cameras).lower(),
            second_enable_cameras=str(second_enable_cameras).lower(),
        ),
    )

    experiment_cfg = _load_experiment(config_path, overrides=overrides)
    composed_requires_cameras = any(run.environment.enable_cameras for run in experiment_cfg.runs.values())

    assert typed_experiment_requires_cameras(config_path, overrides) == composed_requires_cameras
