# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimal Hydra environment-configuration example."""

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.evaluation.arena_experiment_cfg import ArenaExperimentCfg, EnvironmentBuilderCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)
from isaaclab_arena_examples.hydra_configuration.run import (
    _arena_experiment_cfg_to_eval_job,
    _arena_experiment_cfg_to_legacy_builder_namespace,
    _resolve_legacy_eval_runner_namespace,
    compose_from_command_line,
    compose_hydra_example_experiments,
)

EXAMPLE_CONFIG_PATH = (
    Path(__file__).parents[2] / "isaaclab_arena_examples" / "hydra_configuration" / "hydra_example_suite.yaml"
)


def test_hydra_example_experiments_port_variations_and_control():
    experiments = compose_hydra_example_experiments(EXAMPLE_CONFIG_PATH)

    assert [experiment.name for experiment in experiments] == ["variations_demo", "baseline_no_variations"]
    assert all(isinstance(experiment, ArenaExperimentCfg) for experiment in experiments)
    assert all(isinstance(experiment.environment, PickAndPlaceMapleTableEnvironmentCfg) for experiment in experiments)
    assert all(experiment.environment.name == "pick_and_place_maple_table" for experiment in experiments)
    assert all(experiment.environment.enable_cameras for experiment in experiments)
    assert all(experiment.environment.embodiment_asset_name == "droid_rel_joint_pos" for experiment in experiments)
    assert all(
        experiment.environment.high_dynamic_range_image_name == "home_office_robolab" for experiment in experiments
    )
    assert all(
        experiment.environment.pick_up_object_asset_name == "rubiks_cube_hot3d_robolab" for experiment in experiments
    )
    assert all(
        experiment.environment.destination_location_asset_name == "bowl_ycb_robolab" for experiment in experiments
    )
    assert experiments[0].environment is not experiments[1].environment
    assert all(experiment.policy.type == "zero_action" for experiment in experiments)
    assert all(experiment.rollout.num_steps == 10 for experiment in experiments)
    assert all(experiment.num_rebuilds == 1 for experiment in experiments)
    assert experiments[0].variations == {
        "light.hdr_image.enabled": True,
        "light.intensity.enabled": True,
        "droid_rel_joint_pos.camera_extrinsics_wrist_camera.enabled": True,
    }
    assert experiments[1].variations == {}


def test_hydra_example_experiments_accept_typed_shared_environment_override():
    experiments = compose_hydra_example_experiments(EXAMPLE_CONFIG_PATH, ["environment.light_intensity=825"])

    assert [experiment.environment.light_intensity for experiment in experiments] == [825.0, 825.0]


def test_hydra_example_experiments_preserve_add_operator_for_shared_override():
    experiments = compose_hydra_example_experiments(EXAMPLE_CONFIG_PATH, ["+policy.parameters.checkpoint=/tmp/model"])

    assert [experiment.policy.parameters["checkpoint"] for experiment in experiments] == [
        "/tmp/model",
        "/tmp/model",
    ]


def test_hydra_example_cli_keeps_dispatcher_flags_out_of_experiment_configuration():
    experiments, dispatcher_arguments = compose_from_command_line([
        str(EXAMPLE_CONFIG_PATH),
        "--device",
        "cuda:1",
        "--viz",
        "kit",
        "rollout.num_steps=1",
    ])
    eval_arguments = _resolve_legacy_eval_runner_namespace(
        experiments,
        dispatcher_arguments.device,
        dispatcher_arguments.visualizer,
    )

    assert dispatcher_arguments.device == "cuda:1"
    assert dispatcher_arguments.visualizer == "kit"
    assert [experiment.rollout.num_steps for experiment in experiments] == [1, 1]
    assert eval_arguments.device == "cuda:1"
    assert eval_arguments.visualizer == "kit"
    assert eval_arguments.enable_cameras


def test_hydra_example_cli_requires_a_yaml_path():
    with pytest.raises(SystemExit):
        compose_from_command_line([])


def test_dispatcher_aggregates_camera_requirements_without_mutating_experiments():
    no_cameras = ArenaExperimentCfg(
        name="no_cameras",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
    )
    with_cameras = ArenaExperimentCfg(
        name="with_cameras",
        environment=PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True),
    )

    eval_arguments = _resolve_legacy_eval_runner_namespace([no_cameras, with_cameras], "cuda:0", None)

    assert eval_arguments.enable_cameras
    assert not no_cameras.environment.enable_cameras
    assert with_cameras.environment.enable_cameras


def test_hydra_example_experiments_reject_unknown_environment_option():
    with pytest.raises(ConfigCompositionException, match="unknown_option"):
        compose_hydra_example_experiments(EXAMPLE_CONFIG_PATH, ["environment.unknown_option=true"])


def test_hydra_example_experiments_apply_python_defaults_to_minimal_yaml(tmp_path):
    config_path = tmp_path / "minimal_experiments.yaml"
    config_path.write_text("""\
experiments:
  - name: maple_table_experiment
    environment:
      name: pick_and_place_maple_table
""")

    experiments = compose_hydra_example_experiments(config_path)

    assert [experiment.name for experiment in experiments] == ["maple_table_experiment"]
    assert isinstance(experiments[0].environment, PickAndPlaceMapleTableEnvironmentCfg)


def test_hydra_example_experiments_resolve_environment_configuration_through_registry():
    experiments = compose_hydra_example_experiments(EXAMPLE_CONFIG_PATH)

    provider = EnvironmentRegistry().get_component_by_name(experiments[0].environment.name)

    assert provider is PickAndPlaceMapleTableEnvironment
    assert provider.cfg_type is PickAndPlaceMapleTableEnvironmentCfg


def test_hydra_example_experiments_require_an_environment_name(tmp_path):
    config_path = tmp_path / "missing_environment_name.yaml"
    config_path.write_text("""\
experiments:
  - name: maple_table_experiment
    environment: {}
""")

    with pytest.raises(AssertionError, match="environment.name must be a string"):
        compose_hydra_example_experiments(config_path)


def test_hydra_example_experiments_reject_unknown_yaml_environment_option(tmp_path):
    config_path = tmp_path / "invalid_experiments.yaml"
    config_path.write_text("""\
experiments:
  - name: maple_table_experiment
    environment:
      name: pick_and_place_maple_table
      unknown_option: true
""")

    with pytest.raises(ConfigCompositionException, match="unknown_option"):
        compose_hydra_example_experiments(config_path)


def test_hydra_example_experiments_reject_duplicate_names(tmp_path):
    config_path = tmp_path / "duplicate_experiments.yaml"
    config_path.write_text("""\
experiments:
  - name: duplicate
    environment:
      name: pick_and_place_maple_table
  - name: duplicate
    environment:
      name: pick_and_place_maple_table
""")

    with pytest.raises(AssertionError, match="experiment names must be unique"):
        compose_hydra_example_experiments(config_path)


def test_arena_experiment_cfg_maps_to_legacy_builder_namespace():
    experiment = ArenaExperimentCfg(
        name="builder_arguments",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
        environment_builder=EnvironmentBuilderCfg(
            num_envs=3,
            env_spacing=12.5,
            seed=7,
            solve_relations=False,
            placement_seed=9,
            resolve_on_reset=False,
            random_yaw_init=True,
            disable_fabric=True,
            mimic=True,
            presets="layout.yaml",
        ),
    )

    namespace = _arena_experiment_cfg_to_legacy_builder_namespace(experiment, "cuda:2", "pick up the cube")

    assert vars(namespace) == {
        "num_envs": 3,
        "env_spacing": 12.5,
        "seed": 7,
        "solve_relations": False,
        "placement_seed": 9,
        "resolve_on_reset": False,
        "random_yaw_init": True,
        "disable_fabric": True,
        "mimic": True,
        "presets": "layout.yaml",
        "device": "cuda:2",
        "language_instruction": "pick up the cube",
    }


def test_hydra_experiments_map_to_eval_jobs_without_environment_cli_round_trip():
    experiments = compose_hydra_example_experiments(
        EXAMPLE_CONFIG_PATH,
        [
            "environment_builder.num_envs=3",
            "policy.type=zero_action",
            "rollout.num_steps=7",
        ],
    )

    eval_jobs = [_arena_experiment_cfg_to_eval_job(experiment) for experiment in experiments]

    assert [job.name for job in eval_jobs] == ["variations_demo", "baseline_no_variations"]
    assert all(job.num_envs == 3 for job in eval_jobs)
    assert all(job.num_steps == 7 for job in eval_jobs)
    assert all(job.num_rebuilds == 1 for job in eval_jobs)
    assert all(job.policy_type == "zero_action" for job in eval_jobs)
    assert all(job.policy_config_dict == {} for job in eval_jobs)
    assert all(job.arena_env_args == [] for job in eval_jobs)
    assert set(eval_jobs[0].variations) == {
        "light.hdr_image.enabled=true",
        "light.intensity.enabled=true",
        "droid_rel_joint_pos.camera_extrinsics_wrist_camera.enabled=true",
    }
    assert eval_jobs[1].variations == []
