# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test structured composition of keyed Arena experiment collections."""

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from isaaclab_arena.evaluation.experiment_collection_hydra import compose_experiment_collection
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

_ENVIRONMENT_CFG_TYPES = {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg}
_POLICY_CFG_TYPES = {"zero_action": ZeroActionPolicyCfg}


def _compose(config_path: Path, overrides: list[str] | None = None):
    return compose_experiment_collection(
        config_path,
        environment_cfg_types=_ENVIRONMENT_CFG_TYPES,
        policy_cfg_types=_POLICY_CFG_TYPES,
        overrides=overrides,
    )


def test_cfg_defaults_then_yaml_then_cli_override():
    config_path = Path(TestConstants.arena_environments_dir) / "experiment_configs" / "droid_pnp_variations_config.yaml"

    collection = _compose(
        config_path,
        overrides=["experiments.variations_demo.environment.light_intensity=750"],
    )

    assert list(collection.experiments) == ["variations_demo"]
    experiment = collection.experiments["variations_demo"]
    assert experiment.name == "variations_demo"
    assert experiment.environment == PickAndPlaceMapleTableEnvironmentCfg(
        enable_cameras=True,
        embodiment="droid_rel_joint_pos",
        hdr="home_office_robolab",
        light_intensity=750.0,
    )
    assert experiment.policy == ZeroActionPolicyCfg()
    assert experiment.rollout.num_steps == 10
    assert experiment.num_rebuilds == 1


def test_keyed_experiments_preserve_yaml_order(tmp_path):
    config_path = tmp_path / "ordered.yaml"
    config_path.write_text(
        """\
experiments:
  first:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
    rollout:
      num_steps: 1
  second:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
    rollout:
      num_steps: 1
""",
        encoding="utf-8",
    )

    collection = _compose(config_path)

    assert list(collection.experiments) == ["first", "second"]
    assert [experiment.name for experiment in collection.experiments.values()] == ["first", "second"]


def test_unknown_concrete_config_field_is_rejected(tmp_path):
    config_path = tmp_path / "unknown_field.yaml"
    config_path.write_text(
        """\
experiments:
  invalid:
    environment:
      type: pick_and_place_maple_table
      unknown_field: value
    policy:
      type: zero_action
    rollout:
      num_steps: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigCompositionException, match="unknown_field"):
        _compose(config_path)


def test_collection_and_type_selectors_are_required(tmp_path):
    empty_config_path = tmp_path / "empty.yaml"
    empty_config_path.write_text("experiments: {}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="must not be empty"):
        _compose(empty_config_path)

    missing_type_path = tmp_path / "missing_type.yaml"
    missing_type_path.write_text(
        """\
experiments:
  invalid:
    environment: {}
    policy:
      type: zero_action
    rollout:
      num_steps: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="environment.type is required"):
        _compose(missing_type_path)
