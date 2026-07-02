# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimal Hydra environment-configuration example."""

import pytest
from hydra.errors import ConfigCompositionException

from isaaclab_arena_examples.hydra_configuration.config import ArenaRunConfiguration, compose_hydra_example_suite
from isaaclab_arena_examples.hydra_configuration.pick_and_place_maple_table import (
    PickAndPlaceMapleTableEnvironmentConfiguration,
)
from isaaclab_arena_examples.hydra_configuration.run import _job_from_configuration, compose_from_command_line


def test_hydra_example_suite_composes_to_concrete_environment_configuration():
    configuration = compose_hydra_example_suite()

    assert isinstance(configuration, ArenaRunConfiguration)
    assert isinstance(configuration.environment, PickAndPlaceMapleTableEnvironmentConfiguration)
    assert configuration.environment.embodiment_asset_name == "droid_abs_joint_pos"
    assert configuration.environment.pick_up_object_asset_name == "rubiks_cube_hot3d_robolab"
    assert configuration.policy.type == "zero_action"
    assert configuration.rollout.num_steps > 0


def test_hydra_example_suite_accepts_typed_environment_override():
    configuration = compose_hydra_example_suite(["environment.light_intensity=750"])

    assert configuration.environment.light_intensity == 750.0


def test_hydra_example_cli_maps_visualizer_and_forwards_hydra_overrides():
    configuration = compose_from_command_line(["--viz", "kit", "rollout.num_steps=1"])

    assert configuration.simulation_app.visualizer == "kit"
    assert configuration.rollout.num_steps == 1


def test_hydra_example_suite_rejects_unknown_environment_option():
    with pytest.raises(ConfigCompositionException, match="unknown_option"):
        compose_hydra_example_suite(["environment.unknown_option=true"])


def test_hydra_run_maps_to_eval_job_without_environment_cli_round_trip():
    configuration = compose_hydra_example_suite([
        "name=mapped_run",
        "environment_builder.num_envs=3",
        "policy.type=zero_action",
        "rollout.num_steps=7",
    ])

    job = _job_from_configuration(configuration)

    assert job.name == "mapped_run"
    assert job.num_envs == 3
    assert job.num_steps == 7
    assert job.policy_type == "zero_action"
    assert job.policy_config_dict == {}
    assert job.arena_env_args == []
