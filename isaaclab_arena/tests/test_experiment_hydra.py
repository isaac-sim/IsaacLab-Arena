# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test Hydra composition of typed YAML Experiments."""

import yaml
from pathlib import Path
from yaml.constructor import ConstructorError

import pytest
from hydra import initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import RolloutLimitCfg
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.hydra.typed_experiment_loader import (
    load_arena_experiment_from_yaml,
    load_experiment_run_definitions_from_yaml,
    split_shared_run_default_overrides,
)
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

GETTING_STARTED_EXPERIMENT_PATH = (
    Path(TestConstants.arena_environments_dir) / "experiment_configs" / "getting_started_experiment.yaml"
)
ROBOLAB_EXPERIMENT_CONFIG_DIRECTORY = Path(TestConstants.arena_environments_dir) / "robolab" / "experiment_configs"


def _policy_cfg_type_for_name_or_class_path(policy_name_or_class_path: str) -> type[ZeroActionPolicyCfg]:
    return {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path]


def _load_experiment(config_path: str | Path, overrides: list[str] | None = None):
    return load_arena_experiment_from_yaml(
        config_path,
        environment_cfg_types={"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
        policy_cfg_type_resolver=_policy_cfg_type_for_name_or_class_path,
        overrides=overrides,
    )


def _write_experiment(tmp_path: Path, contents: str) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_getting_started_experiment_composes_typed_runs():
    experiment_cfg = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH)
    runs = experiment_cfg.runs

    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    assert list(runs) == [
        "baseline",
        "swap_objects",
        "change_background_hdr",
        "parallel_envs",
    ]
    assert runs["baseline"].environment == PickAndPlaceMapleTableEnvironmentCfg(
        embodiment="droid_rel_joint_pos",
        hdr="home_office_robolab",
        episode_length_s=1.5,
    )
    assert runs["swap_objects"].environment == PickAndPlaceMapleTableEnvironmentCfg(
        embodiment="droid_rel_joint_pos",
        destination_location="wooden_bowl_hot3d_robolab",
        hdr="home_office_robolab",
        episode_length_s=1.5,
    )
    assert runs["swap_objects"].num_rebuilds == 3
    assert runs["swap_objects"].variations == {
        "pick_up_object.asset_selection.enabled": True,
        "pick_up_object.asset_selection.registry_names": [
            "mustard_bottle_hot3d_robolab",
            "orange_01_fruits_veggies_robolab",
            "rubiks_cube_hot3d_robolab",
        ],
    }
    assert runs["change_background_hdr"].environment.hdr == "billiard_hall_robolab"
    assert runs["parallel_envs"].environment.hdr == "home_office_robolab"
    assert all(run.environment.episode_length_s == 1.5 for run in runs.values())
    assert all(run.policy == ZeroActionPolicyCfg() for run in runs.values())
    assert [run.environment_builder.num_envs for run in runs.values()] == [1, 1, 1, 64]
    assert runs["parallel_envs"].environment_builder.env_spacing == 2.5
    assert all(run.rollout_limit.num_steps is None for run in runs.values())
    assert [run.rollout_limit.num_episodes for run in runs.values()] == [1, 3, 1, 1]


def test_experiment_composition_preserves_caller_owned_hydra_context():
    with initialize(version_base=None, config_path=None):
        caller_global_hydra = GlobalHydra.instance()
        caller_hydra = caller_global_hydra.hydra

        _load_experiment(GETTING_STARTED_EXPERIMENT_PATH)

        assert GlobalHydra.instance() is caller_global_hydra
        assert GlobalHydra.instance().hydra is caller_hydra


def test_repeated_composition_reuses_config_store_entries(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  first:
    environment:
      type: pick_and_place_maple_table
      light_intensity: 600.0
    policy:
      type: zero_action
""",
    )

    with initialize(version_base=None, config_path=None):
        first_experiment = _load_experiment(config_path)
        config_store_names_after_first_load = set(ConfigStore.instance().repo)

        _write_experiment(
            tmp_path,
            """
runs:
  second:
    environment:
      type: pick_and_place_maple_table
      light_intensity: 700.0
    policy:
      type: zero_action
""",
        )
        second_experiment = _load_experiment(config_path)

        assert set(ConfigStore.instance().repo) == config_store_names_after_first_load

    assert first_experiment.runs["first"].environment.light_intensity == 600.0
    assert second_experiment.runs["second"].environment.light_intensity == 700.0


def _test_getting_started_experiment_executes_baseline_run(simulation_app, output_dir: Path):
    """Load the checked-in Experiment and execute its baseline Run in Isaac Sim."""
    from isaaclab_arena.evaluation.run_execution import build_and_run
    from isaaclab_arena.evaluation.run_status import RunStatus

    baseline_run = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH).runs["baseline"]

    result = build_and_run(baseline_run, output_dir=output_dir)

    assert result.run_name == "baseline"
    assert result.status is RunStatus.COMPLETED
    return True


def test_getting_started_experiment_executes_baseline_run(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_getting_started_experiment_executes_baseline_run,
        headless=True,
        output_dir=tmp_path,
    )


def test_runs_keep_yaml_order_and_hydra_overrides_take_precedence(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  first:
    environment:
      type: pick_and_place_maple_table
      light_intensity: 600.0
    policy:
      type: zero_action
  second:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""",
    )

    experiment_cfg = _load_experiment(
        config_path,
        overrides=[
            "runs.first.environment.light_intensity=750.0",
            "runs.first.environment_builder.record_trajectories=true",
            "runs.second.environment.enable_cameras=true",
        ],
    )

    assert list(experiment_cfg.runs) == ["first", "second"]
    assert experiment_cfg.runs["first"].environment.light_intensity == 750.0
    assert experiment_cfg.runs["first"].environment_builder.record_trajectories is True
    assert experiment_cfg.runs["second"].environment.enable_cameras is True
    assert experiment_cfg.runs["second"].environment_builder.record_trajectories is False


def test_shared_run_defaults_merge_before_run_overrides_without_resolving_local_interpolations(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
shared:
  environment:
    light_intensity: 600.0
  environment_builder:
    num_envs: 2
  variations:
    camera:
      enabled: false
  rollout_limit:
    num_steps: ${environment_builder.num_envs}

runs:
  first:
    environment:
      type: pick_and_place_maple_table
      destination_location: ${pick_up_object}
    policy:
      type: zero_action
  second:
    environment:
      type: pick_and_place_maple_table
      light_intensity: 650.0
    policy:
      type: zero_action
""",
    )

    experiment_cfg = _load_experiment(
        config_path,
        overrides=[
            "shared.environment.light_intensity=750.0",
            "shared.variations.camera.enabled=true",
            "runs.second.variations.camera.enabled=false",
        ],
    )

    first_run = experiment_cfg.runs["first"]
    second_run = experiment_cfg.runs["second"]
    assert first_run.environment.light_intensity == 750.0
    assert first_run.environment.destination_location == first_run.environment.pick_up_object
    assert first_run.rollout_limit.num_steps == first_run.environment_builder.num_envs == 2
    assert first_run.variations["camera"]["enabled"] is True
    assert second_run.environment.light_intensity == 650.0
    assert second_run.environment_builder.num_envs == 2
    assert second_run.variations["camera"]["enabled"] is False
    with pytest.raises(ValueError, match="missing"):
        _load_experiment(config_path, overrides=["shared.missing=true"])


def test_run_rollout_limit_replaces_shared_rollout_limit(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
shared:
  environment:
    type: pick_and_place_maple_table
  policy:
    type: zero_action
  rollout_limit:
    num_steps: 50

runs:
  inherited: {}
  total_episodes:
    rollout_limit:
      num_episodes: 4
  per_rebuild:
    num_rebuilds: 2
    rollout_limit:
      num_episodes_per_rebuild: [3, 1]
""",
    )

    runs = _load_experiment(config_path).runs
    assert runs["inherited"].rollout_limit == RolloutLimitCfg(num_steps=50)
    assert runs["total_episodes"].rollout_limit == RolloutLimitCfg(num_episodes=4)
    assert runs["per_rebuild"].rollout_limit == RolloutLimitCfg(num_episodes_per_rebuild=[3, 1])


@pytest.mark.parametrize(("operator", "assignment"), [("+", "=75"), ("++", "=75"), ("~", "")])
@pytest.mark.parametrize("experiment_config_prefix", ["", "experiment_cfg"])
def test_shared_run_default_overrides_reject_hydra_operators(operator, assignment, experiment_config_prefix):
    override_prefix = f"{experiment_config_prefix}." if experiment_config_prefix else ""
    override = f"{operator}{override_prefix}shared.rollout_limit.num_steps{assignment}"

    with pytest.raises(AssertionError, match="uses an unsupported Hydra operator"):
        split_shared_run_default_overrides([override], experiment_config_prefix)


@pytest.mark.parametrize(
    ("config_name", "expected_run_count"),
    [
        ("robolab_18_tasks_pi0_and_cosmos.yaml", 36),
        ("robolab_20_tasks_pi0_and_cosmos.yaml", 40),
    ],
)
def test_robolab_experiments_apply_shared_run_defaults(config_name, expected_run_count):
    config_path = ROBOLAB_EXPERIMENT_CONFIG_DIRECTORY / config_name

    default_run_values = load_experiment_run_definitions_from_yaml(config_path)
    enabled_run_values = load_experiment_run_definitions_from_yaml(
        config_path,
        shared_default_overrides=["shared.variations.droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true"],
    )

    assert len(default_run_values) == expected_run_count
    assert len(enabled_run_values) == expected_run_count
    assert all(run["environment"]["enable_cameras"] for run in default_run_values.values())
    assert all(run["environment_builder"]["num_envs"] == 20 for run in default_run_values.values())
    assert all(run["rollout_limit"]["num_episodes"] == 100 for run in default_run_values.values())
    assert not any(
        run["variations"]["droid_abs_joint_pos"]["camera_extrinsics_wrist_camera"]["enabled"]
        for run in default_run_values.values()
    )
    assert all(
        run["variations"]["droid_abs_joint_pos"]["camera_extrinsics_wrist_camera"]["enabled"]
        for run in enabled_run_values.values()
    )


def test_effective_experiment_serializes_to_reloadable_yaml(tmp_path):
    experiment_cfg = _load_experiment(
        GETTING_STARTED_EXPERIMENT_PATH,
        overrides=["runs.baseline.environment.light_intensity=750.0"],
    )

    serialized_experiment = serialize_arena_experiment_to_yaml(experiment_cfg)
    serialized_values = yaml.safe_load(serialized_experiment)
    serialized_baseline = serialized_values["runs"]["baseline"]
    assert serialized_baseline["environment"]["type"] == "pick_and_place_maple_table"
    assert serialized_baseline["policy"]["type"] == "zero_action"
    assert serialized_baseline["environment"]["light_intensity"] == 750.0
    assert "name" not in serialized_baseline

    serialized_path = _write_experiment(tmp_path, serialized_experiment)
    assert _load_experiment(serialized_path) == experiment_cfg


def test_serializer_round_trips_the_name(tmp_path):
    experiment_cfg = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH, overrides=["name=my-benchmark"])
    assert experiment_cfg.name == "my-benchmark"

    serialized_experiment = serialize_arena_experiment_to_yaml(experiment_cfg)

    assert list(yaml.safe_load(serialized_experiment)) == ["name", "kpi_exporters", "runs"]
    serialized_path = _write_experiment(tmp_path, serialized_experiment)
    assert _load_experiment(serialized_path) == experiment_cfg


def test_yaml_without_a_name_defaults_to_the_file_stem():
    experiment_cfg = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH)

    assert experiment_cfg.name == GETTING_STARTED_EXPERIMENT_PATH.stem


def test_serializer_writes_the_name():
    experiment_cfg = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH)

    assert yaml.safe_load(serialize_arena_experiment_to_yaml(experiment_cfg))["name"] == experiment_cfg.name


_MINIMAL_RUN_YAML = """runs:
  baseline:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
"""


def test_yaml_without_kpi_exporters_defaults_to_timing(tmp_path):
    experiment_cfg = _load_experiment(_write_experiment(tmp_path, _MINIMAL_RUN_YAML))

    assert experiment_cfg.kpi_exporters == ["timing"]


def test_yaml_top_level_kpi_exporters_selects_the_exporters(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: [success_rate, timing]\n" + _MINIMAL_RUN_YAML)

    assert _load_experiment(config_path).kpi_exporters == ["success_rate", "timing"]


def test_yaml_empty_kpi_exporters_exports_no_kpis(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: []\n" + _MINIMAL_RUN_YAML)

    assert _load_experiment(config_path).kpi_exporters == []


def test_hydra_override_replaces_kpi_exporters(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: [timing]\n" + _MINIMAL_RUN_YAML)

    experiment_cfg = _load_experiment(config_path, overrides=["kpi_exporters=[success_rate,timing]"])

    assert experiment_cfg.kpi_exporters == ["success_rate", "timing"]


def test_hydra_override_can_clear_kpi_exporters(tmp_path):
    experiment_cfg = _load_experiment(_write_experiment(tmp_path, _MINIMAL_RUN_YAML), overrides=["kpi_exporters=[]"])

    assert experiment_cfg.kpi_exporters == []


def test_unknown_kpi_exporter_in_yaml_is_rejected(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: [latency]\n" + _MINIMAL_RUN_YAML)

    with pytest.raises(AssertionError, match="Unknown KPI exporters"):
        _load_experiment(config_path)


def test_kpi_exporters_must_be_a_list(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: timing\n" + _MINIMAL_RUN_YAML)

    with pytest.raises(AssertionError, match="'kpi_exporters' must be a list"):
        _load_experiment(config_path)


def test_serializer_round_trips_the_kpi_exporters(tmp_path):
    config_path = _write_experiment(tmp_path, "kpi_exporters: [success_rate, timing]\n" + _MINIMAL_RUN_YAML)
    experiment_cfg = _load_experiment(config_path)

    serialized_experiment = serialize_arena_experiment_to_yaml(experiment_cfg)

    assert yaml.safe_load(serialized_experiment)["kpi_exporters"] == ["success_rate", "timing"]
    assert _load_experiment(_write_experiment(tmp_path, serialized_experiment)) == experiment_cfg


def test_replay_paths_round_trip_and_infer_rebuild_count(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  replay:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
    recorded_variation_samples_paths:
      - recordings/rebuild0.jsonl
      - recordings/rebuild1.jsonl
""",
    )

    experiment_cfg = _load_experiment(config_path)
    run = experiment_cfg.runs["replay"]
    assert run.recorded_variation_samples_paths == [
        "recordings/rebuild0.jsonl",
        "recordings/rebuild1.jsonl",
    ]
    assert run.num_rebuilds == 2

    serialized_path = _write_experiment(tmp_path, serialize_arena_experiment_to_yaml(experiment_cfg))
    assert _load_experiment(serialized_path) == experiment_cfg


def test_episodes_per_rebuild_round_trip_and_accept_matching_overrides(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  per_rebuild:
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
    num_rebuilds: 3
    rollout_limit:
      num_episodes_per_rebuild: [20, 5, 5]
""",
    )

    experiment_cfg = _load_experiment(config_path)
    run = experiment_cfg.runs["per_rebuild"]
    assert run.rollout_limit.num_episodes_per_rebuild == [20, 5, 5]

    overridden_cfg = _load_experiment(
        config_path,
        overrides=[
            "runs.per_rebuild.num_rebuilds=2",
            "runs.per_rebuild.rollout_limit.num_episodes_per_rebuild=[4,2]",
        ],
    )
    assert overridden_cfg.runs["per_rebuild"].num_rebuilds == 2
    assert overridden_cfg.runs["per_rebuild"].rollout_limit.num_episodes_per_rebuild == [4, 2]

    serialized_path = _write_experiment(tmp_path, serialize_arena_experiment_to_yaml(experiment_cfg))
    assert _load_experiment(serialized_path) == experiment_cfg


GRAPH_SPEC_EXPERIMENT_CONTENTS = """
runs:
  graph_run:
    environment:
      type: robolab/tasks/banana_in_bowl.yaml
      pick_up_object: banana
    policy:
      type: zero_action
    environment_builder:
      num_envs: 2
    rollout_limit:
      num_steps: 5
"""


def test_graph_spec_yaml_environment_builds_legacy_cfg(tmp_path):
    config_path = _write_experiment(tmp_path, GRAPH_SPEC_EXPERIMENT_CONTENTS)

    experiment_cfg = _load_experiment(config_path)
    run = experiment_cfg.runs["graph_run"]

    assert run.environment == LegacyGraphEnvironmentCfg(
        env_spec_path="robolab/tasks/banana_in_bowl.yaml",
        per_run_overrides={"pick_up_object": "banana"},
    )
    assert run.policy == ZeroActionPolicyCfg()
    assert run.environment_builder.num_envs == 2
    assert run.rollout_limit.num_steps == 5


@pytest.mark.parametrize(
    ("config_name", "run_name", "num_envs"),
    [
        ("recorded_variations_workflow_experiment.yaml", "recorded_variations", 4),
        ("recorded_variations_replay_experiment.yaml", "recorded_variations_replay", 3),
    ],
)
def test_recorded_variations_workflow_accepts_cli_timeout_override(config_name, run_name, num_envs):
    config_path = (
        Path(TestConstants.arena_environments_dir) / "experiment_configs" / "placement_record_replay" / config_name
    )
    baseline = _load_experiment(config_path).runs[run_name]
    assert baseline.env_cfg_override == {}

    run = _load_experiment(config_path, overrides=[f"+runs.{run_name}.env_cfg_override={{episode_length_s:1.0}}"]).runs[
        run_name
    ]
    assert run.env_cfg_override == {"episode_length_s": 1.0}
    assert run.environment == baseline.environment
    assert run.environment_builder == baseline.environment_builder
    assert run.environment_builder.num_envs == num_envs
    assert run.rollout_limit.num_episodes == 128
    assert run.num_rebuilds == 2


def _test_run_env_cfg_override_merges_experiment_defaults_and_round_trips(simulation_app, tmp_path, environment_type):
    contents = yaml.safe_load(GRAPH_SPEC_EXPERIMENT_CONTENTS)
    if environment_type == "python":
        contents["runs"]["graph_run"]["environment"] = {"type": "pick_and_place_maple_table"}
    contents["shared"] = {"env_cfg_override": {"episode_length_s": 4.0, "sim": {"dt": 0.01}}}
    contents["runs"]["graph_run"]["env_cfg_override"] = {"episode_length_s": 2.0}
    config_path = _write_experiment(tmp_path, yaml.safe_dump(contents))
    assert _load_experiment(config_path).runs["graph_run"].env_cfg_override == {
        "episode_length_s": 2.0,
        "sim": {"dt": 0.01},
    }

    experiment = _load_experiment(config_path, overrides=["runs.graph_run.env_cfg_override.episode_length_s=1.0"])
    run = experiment.runs["graph_run"]
    assert run.env_cfg_override == {"episode_length_s": 1.0, "sim": {"dt": 0.01}}
    serialized = serialize_arena_experiment_to_yaml(experiment)
    serialized_run = yaml.safe_load(serialized)["runs"]["graph_run"]
    assert serialized_run["env_cfg_override"] == run.env_cfg_override
    assert "env_cfg_override" not in serialized_run["environment"]
    assert _load_experiment(_write_experiment(tmp_path, serialized)) == experiment
    return True


@pytest.mark.parametrize("environment_type", ["graph", "python"])
def test_run_env_cfg_override_merges_experiment_defaults_and_round_trips(tmp_path, environment_type):
    # Serialization loads asset registries, so it must run after SimulationApp starts.
    assert run_function_with_persistent_simulation_app(
        _test_run_env_cfg_override_merges_experiment_defaults_and_round_trips,
        tmp_path=tmp_path,
        environment_type=environment_type,
    )


def _test_run_env_cfg_override_applies_after_environment_callback(simulation_app, environment_type):
    import math
    import sys
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab.envs.mdp import time_out

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    if environment_type == "graph":
        config_path = (
            Path(TestConstants.arena_environments_dir)
            / "experiment_configs/placement_record_replay/recorded_variations_workflow_experiment.yaml"
        )
        run_name = "recorded_variations"
        task_timeout_s = 60.0
    else:
        config_path = GETTING_STARTED_EXPERIMENT_PATH
        run_name = "baseline"
        task_timeout_s = 1.5
    experiment = _load_experiment(
        config_path,
        overrides=[
            f"+runs.{run_name}.env_cfg_override="
            "{episode_length_s:1.0,sim:{physics:{max_position_iteration_count:16}}}"
        ],
    )
    run = experiment.runs[run_name]
    run.environment_builder = ArenaEnvBuilderCfg(solve_relations=False, presets="physx")
    # Placement replay and variations are exercised elsewhere; this checks final config construction.
    run.variations = {}
    argv = ["experiment_runner.py"]
    if environment_type == "graph":
        argv.extend(["--env_spec", run.environment.env_spec_path])
    # Isolate the graph parser from pytest's arguments without invoking an exit path.
    with patch.object(sys, "argv", argv):
        builder = build_arena_builder_from_run_cfg(run)
    env_cfg, _ = builder.build_env_cfg()
    assert env_cfg.episode_length_s == 1.0
    assert env_cfg.sim.physics.max_position_iteration_count == 16
    # Preserve settings from either kind of environment's existing callback.
    if environment_type == "graph":
        assert env_cfg.sim.physics.max_velocity_iteration_count == 1
    else:
        assert env_cfg.viewer.eye == (1.5, 0.0, 1.0)
        assert env_cfg.viewer.lookat == (0.2, 0.0, 0.0)
    assert env_cfg.terminations.time_out.func is time_out
    assert env_cfg.terminations.time_out.time_out
    task_termination = builder.arena_env.task.get_termination_cfg()
    assert task_termination.timeout_s == task_timeout_s
    assert task_termination.success
    assert env_cfg.terminations.success is not None
    assert task_termination.failures
    for name, term in task_termination.failures.items():
        assert getattr(env_cfg.terminations, name).func is term.func

    max_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.decimation * env_cfg.sim.dt))
    env = SimpleNamespace(max_episode_length=max_steps, episode_length_buf=torch.tensor([max_steps - 1, max_steps]))
    assert env_cfg.terminations.time_out.func(env).tolist() == [False, True]
    assert run.env_cfg_override["episode_length_s"] == 1.0

    run.env_cfg_override = {"sim": {"unknown_field": 1}}
    with pytest.raises(ValueError, match="Invalid env_cfg_override"):
        builder.build_env_cfg()
    return True


@pytest.mark.parametrize("environment_type", ["graph", "python"])
def test_run_env_cfg_override_applies_after_environment_callback(environment_type):
    assert run_function_with_persistent_simulation_app(
        _test_run_env_cfg_override_applies_after_environment_callback, environment_type=environment_type
    )


def test_graph_spec_yaml_environment_populates_enable_cameras(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  graph_run:
    environment:
      type: robolab/tasks/banana_in_bowl.yaml
      enable_cameras: true
    policy:
      type: zero_action
""",
    )

    run = _load_experiment(config_path).runs["graph_run"]

    # The typed enable_cameras field drives pre-startup camera detection; the same value is
    # retained in per_run_overrides, from which the graph-environment CLI tokens are built at
    # execution. Both must reflect the YAML value.
    assert run.environment.enable_cameras is True
    assert run.environment.per_run_overrides["enable_cameras"] is True


@pytest.mark.parametrize(
    ("run_contents", "exception_type", "error"),
    [
        (
            """
    environment:
      type: missing_environment
    policy:
      type: zero_action
""",
            AssertionError,
            "unknown environment type 'missing_environment'",
        ),
        (
            """
    environment:
      type: pick_and_place_maple_table
    policy: {}
""",
            AssertionError,
            "missing 'policy.type'",
        ),
        (
            """
    environment:
      type: pick_and_place_maple_table
      unknown_field: true
    policy:
      type: zero_action
""",
            ValueError,
            "unknown_field",
        ),
    ],
)
def test_invalid_run_configuration_is_rejected(tmp_path, run_contents, exception_type, error):
    config_path = _write_experiment(tmp_path, f"runs:\n  invalid:{run_contents}")

    with pytest.raises(exception_type, match=error):
        _load_experiment(config_path)


def test_runs_must_be_a_mapping(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  - name: baseline
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""",
    )

    with pytest.raises(AssertionError, match="must be a mapping from Run names"):
        _load_experiment(config_path)


def test_shared_values_must_be_a_mapping(tmp_path):
    config_path = _write_experiment(tmp_path, "shared: true\nruns:\n  maple_table: {}\n")

    with pytest.raises(AssertionError, match="must be a mapping"):
        _load_experiment(config_path)


def test_experiment_requires_at_least_one_run(tmp_path):
    config_path = _write_experiment(tmp_path, "runs: {}\n")

    with pytest.raises(AssertionError, match="must define at least one Run"):
        _load_experiment(config_path)


def test_run_name_must_be_non_empty(tmp_path):
    config_path = _write_experiment(tmp_path, 'runs:\n  "": {}\n')

    with pytest.raises(AssertionError, match="Run names must be non-empty strings"):
        _load_experiment(config_path)


def test_run_name_must_be_hydra_compatible(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  "01_baseline":
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""",
    )

    with pytest.raises(AssertionError, match="must start with a letter or underscore"):
        _load_experiment(config_path)


def test_duplicate_run_mapping_key_is_rejected(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  maple_table: {}
  maple_table: {}
""",
    )

    with pytest.raises(ConstructorError, match="duplicate key maple_table"):
        _load_experiment(config_path)


def test_run_configuration_must_be_a_mapping(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  maple_table: true
""",
    )

    with pytest.raises(AssertionError, match="Run 'maple_table' must be a mapping"):
        _load_experiment(config_path)


def test_run_must_not_repeat_its_mapping_key_as_name(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        """
runs:
  maple_table:
    name: other_name
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action
""",
    )

    with pytest.raises(AssertionError, match="must not define 'name'"):
        _load_experiment(config_path)


def test_run_mapping_key_cannot_be_overridden(tmp_path):
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

    with pytest.raises(AssertionError, match="cannot be overridden"):
        _load_experiment(config_path, overrides=["runs.maple_table.name=other_name"])


def test_unknown_hydra_override_is_rejected(tmp_path):
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

    with pytest.raises(ValueError, match="unknown_field"):
        _load_experiment(config_path, overrides=["runs.maple_table.environment.unknown_field=true"])


def test_hydra_override_cannot_add_run():
    with pytest.raises(ValueError, match="Error merging override"):
        _load_experiment(
            GETTING_STARTED_EXPERIMENT_PATH,
            overrides=["+runs.new_run={environment:{type:pick_and_place_maple_table},policy:{type:zero_action}}"],
        )
