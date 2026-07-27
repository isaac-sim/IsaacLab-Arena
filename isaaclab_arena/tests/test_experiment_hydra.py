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
from isaaclab_arena.hydra.typed_experiment_loader import load_arena_experiment_from_yaml
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

GETTING_STARTED_EXPERIMENT_PATH = (
    Path(TestConstants.arena_environments_dir) / "experiment_configs" / "getting_started_experiment.yaml"
)


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
    )
    assert runs["swap_objects"].environment == PickAndPlaceMapleTableEnvironmentCfg(
        embodiment="droid_rel_joint_pos",
        pick_up_object="mustard_bottle_hot3d_robolab",
        destination_location="wooden_bowl_hot3d_robolab",
        hdr="home_office_robolab",
    )
    assert runs["change_background_hdr"].environment.hdr == "billiard_hall_robolab"
    assert all(run.policy == ZeroActionPolicyCfg() for run in runs.values())
    assert [run.environment_builder.num_envs for run in runs.values()] == [1, 1, 1, 64]
    assert runs["parallel_envs"].environment_builder.env_spacing == 2.5
    assert [run.rollout_limit.num_steps for run in runs.values()] == [50, 50, 50, 100]


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
    from isaaclab_arena.evaluation.arena_run import RunStatus
    from isaaclab_arena.evaluation.run_execution import build_and_run

    baseline_run = _load_experiment(GETTING_STARTED_EXPERIMENT_PATH).runs["baseline"]

    result = build_and_run(baseline_run, output_dir=output_dir)

    assert result.run_name == "baseline"
    assert result.status is RunStatus.COMPLETED
    return True


def test_getting_started_experiment_executes_baseline_run(tmp_path):
    assert run_simulation_app_function(
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
            "runs.second.environment.enable_cameras=true",
        ],
    )

    assert list(experiment_cfg.runs) == ["first", "second"]
    assert experiment_cfg.runs["first"].environment.light_intensity == 750.0
    assert experiment_cfg.runs["second"].environment.enable_cameras is True


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
