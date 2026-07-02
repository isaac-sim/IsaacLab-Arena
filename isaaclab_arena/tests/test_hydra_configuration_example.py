# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimal Hydra environment-configuration example."""

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.evaluation.arena_job_cfg import ArenaJobCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)
from isaaclab_arena_examples.hydra_configuration.run import (
    _evaluation_runner_arguments,
    _job_from_configuration,
    compose_from_command_line,
    compose_hydra_example_jobs,
)

EXAMPLE_CONFIG_PATH = (
    Path(__file__).parents[2] / "isaaclab_arena_examples" / "hydra_configuration" / "hydra_example_suite.yaml"
)


def test_hydra_example_jobs_port_variations_job_and_control():
    jobs = compose_hydra_example_jobs(EXAMPLE_CONFIG_PATH)

    assert [job.name for job in jobs] == ["variations_demo", "baseline_no_variations"]
    assert all(isinstance(job, ArenaJobCfg) for job in jobs)
    assert all(isinstance(job.environment, PickAndPlaceMapleTableEnvironmentCfg) for job in jobs)
    assert all(job.environment.name == "pick_and_place_maple_table" for job in jobs)
    assert all(job.environment.enable_cameras for job in jobs)
    assert all(job.environment.embodiment_asset_name == "droid_rel_joint_pos" for job in jobs)
    assert all(job.environment.high_dynamic_range_image_name == "home_office_robolab" for job in jobs)
    assert all(job.environment.pick_up_object_asset_name == "rubiks_cube_hot3d_robolab" for job in jobs)
    assert all(job.environment.destination_location_asset_name == "bowl_ycb_robolab" for job in jobs)
    assert jobs[0].environment is not jobs[1].environment
    assert all(job.policy.type == "zero_action" for job in jobs)
    assert all(job.rollout.num_steps == 10 for job in jobs)
    assert all(job.num_rebuilds == 1 for job in jobs)
    assert jobs[0].variations == {
        "light": {
            "hdr_image": {"enabled": True},
            "intensity": {"enabled": True},
        },
        "droid_rel_joint_pos": {
            "camera_extrinsics_wrist_camera": {"enabled": True},
        },
    }
    assert jobs[1].variations == {}


def test_hydra_example_jobs_accept_typed_shared_environment_override():
    jobs = compose_hydra_example_jobs(EXAMPLE_CONFIG_PATH, ["environment.light_intensity=825"])

    assert [job.environment.light_intensity for job in jobs] == [825.0, 825.0]


def test_hydra_example_jobs_preserve_add_operator_for_shared_override():
    jobs = compose_hydra_example_jobs(EXAMPLE_CONFIG_PATH, ["+policy.parameters.checkpoint=/tmp/model"])

    assert [job.policy.parameters["checkpoint"] for job in jobs] == ["/tmp/model", "/tmp/model"]


def test_hydra_example_cli_keeps_dispatcher_flags_out_of_job_configuration():
    jobs, launcher_arguments = compose_from_command_line([
        str(EXAMPLE_CONFIG_PATH),
        "--device",
        "cuda:1",
        "--viz",
        "kit",
        "rollout.num_steps=1",
    ])
    eval_arguments = _evaluation_runner_arguments(
        jobs,
        launcher_arguments.device,
        launcher_arguments.visualizer,
    )

    assert launcher_arguments.device == "cuda:1"
    assert launcher_arguments.visualizer == "kit"
    assert [job.rollout.num_steps for job in jobs] == [1, 1]
    assert eval_arguments.device == "cuda:1"
    assert eval_arguments.visualizer == "kit"
    assert eval_arguments.enable_cameras


def test_hydra_example_cli_requires_a_yaml_path():
    with pytest.raises(SystemExit):
        compose_from_command_line([])


def test_dispatcher_aggregates_camera_requirements_without_mutating_jobs():
    no_cameras = ArenaJobCfg(
        name="no_cameras",
        environment=PickAndPlaceMapleTableEnvironmentCfg(),
    )
    with_cameras = ArenaJobCfg(
        name="with_cameras",
        environment=PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True),
    )

    eval_arguments = _evaluation_runner_arguments([no_cameras, with_cameras], "cuda:0", None)

    assert eval_arguments.enable_cameras
    assert not no_cameras.environment.enable_cameras
    assert with_cameras.environment.enable_cameras


def test_hydra_example_jobs_reject_unknown_environment_option():
    with pytest.raises(ConfigCompositionException, match="unknown_option"):
        compose_hydra_example_jobs(EXAMPLE_CONFIG_PATH, ["environment.unknown_option=true"])


def test_hydra_example_jobs_apply_python_defaults_to_minimal_yaml(tmp_path):
    config_path = tmp_path / "minimal_jobs.yaml"
    config_path.write_text("""\
jobs:
  - name: maple_table_job
    environment:
      name: pick_and_place_maple_table
""")

    jobs = compose_hydra_example_jobs(config_path)

    assert [job.name for job in jobs] == ["maple_table_job"]
    assert isinstance(jobs[0].environment, PickAndPlaceMapleTableEnvironmentCfg)


def test_hydra_example_jobs_resolve_environment_configuration_through_registry():
    jobs = compose_hydra_example_jobs(EXAMPLE_CONFIG_PATH)

    provider = EnvironmentRegistry().get_component_by_name(jobs[0].environment.name)

    assert provider is PickAndPlaceMapleTableEnvironment
    assert provider.cfg_type is PickAndPlaceMapleTableEnvironmentCfg


def test_hydra_example_jobs_require_an_environment_name(tmp_path):
    config_path = tmp_path / "missing_environment_name.yaml"
    config_path.write_text("""\
jobs:
  - name: maple_table_job
    environment: {}
""")

    with pytest.raises(AssertionError, match="environment.name must be a string"):
        compose_hydra_example_jobs(config_path)


def test_hydra_example_jobs_reject_unknown_yaml_environment_option(tmp_path):
    config_path = tmp_path / "invalid_jobs.yaml"
    config_path.write_text("""\
jobs:
  - name: maple_table_job
    environment:
      name: pick_and_place_maple_table
      unknown_option: true
""")

    with pytest.raises(ConfigCompositionException, match="unknown_option"):
        compose_hydra_example_jobs(config_path)


def test_hydra_example_jobs_reject_duplicate_names(tmp_path):
    config_path = tmp_path / "duplicate_jobs.yaml"
    config_path.write_text("""\
jobs:
  - name: duplicate
    environment:
      name: pick_and_place_maple_table
  - name: duplicate
    environment:
      name: pick_and_place_maple_table
""")

    with pytest.raises(AssertionError, match="job names must be unique"):
        compose_hydra_example_jobs(config_path)


def test_hydra_jobs_map_to_eval_jobs_without_environment_cli_round_trip():
    configurations = compose_hydra_example_jobs(
        EXAMPLE_CONFIG_PATH,
        [
            "environment_builder.num_envs=3",
            "policy.type=zero_action",
            "rollout.num_steps=7",
        ],
    )

    jobs = [_job_from_configuration(configuration) for configuration in configurations]

    assert [job.name for job in jobs] == ["variations_demo", "baseline_no_variations"]
    assert all(job.num_envs == 3 for job in jobs)
    assert all(job.num_steps == 7 for job in jobs)
    assert all(job.num_rebuilds == 1 for job in jobs)
    assert all(job.policy_type == "zero_action" for job in jobs)
    assert all(job.policy_config_dict == {} for job in jobs)
    assert all(job.arena_env_args == [] for job in jobs)
    assert set(jobs[0].variations) == {
        "light.hdr_image.enabled=true",
        "light.intensity.enabled=true",
        "droid_rel_joint_pos.camera_extrinsics_wrist_camera.enabled=true",
    }
    assert jobs[1].variations == []
