# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest

from isaaclab_arena.hydra.experiment_values import read_experiment_values
from isaaclab_arena.tests.utils.constants import TestConstants

ROBOLAB_EXPERIMENT_CONFIG_DIRECTORY = Path(TestConstants.arena_environments_dir) / "robolab" / "experiment_configs"

# Every Run must select an environment type and a policy type. Tests that are about something else set both here.
RUN_A_TYPE_OVERRIDES = ["runs.a.environment.type=pick_and_place_maple_table", "runs.a.policy.type=zero_action"]
RUN_TYPES_YAML = """
    environment:
      type: pick_and_place_maple_table
    policy:
      type: zero_action"""


def _write_experiment(tmp_path, contents: str):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(contents, encoding="utf-8")
    return experiment_path


def test_sets_nested_value_and_creates_missing_keys(tmp_path):
    experiment_path = _write_experiment(tmp_path, f"runs:\n  a:{RUN_TYPES_YAML}\n")

    result = read_experiment_values(experiment_path, ["runs.a.environment_builder.num_envs=8"])

    assert result["runs"]["a"]["environment_builder"] == {"num_envs": 8}


def test_without_a_file_overrides_create_runs():
    result = read_experiment_values(
        None,
        [
            "runs.quick.environment.type=kitchen_pick_and_place",
            "runs.quick.policy.type=zero_action",
            "runs.quick.rollout_limit.num_steps=50",
        ],
    )
    assert result["runs"] == {
        "quick": {
            "environment": {"type": "kitchen_pick_and_place"},
            "policy": {"type": "zero_action"},
            "rollout_limit": {"num_steps": 50},
        }
    }


def test_override_values_are_parsed_and_interpolations_stay_unresolved():
    result = read_experiment_values(
        None,
        [
            *RUN_A_TYPE_OVERRIDES,
            "runs.a.list=[a,b]",
            "runs.a.nothing=null",
            "runs.a.mapping={episode_length_s:1.0}",
            "runs.a.rollout_limit.num_steps=${environment_builder.num_envs}",
        ],
    )
    run_values = result["runs"]["a"]
    assert run_values["list"] == ["a", "b"]
    assert run_values["nothing"] is None
    assert run_values["mapping"] == {"episode_length_s": 1.0}
    assert run_values["rollout_limit"]["num_steps"] == "${environment_builder.num_envs}"


def test_plus_operators_mean_plain_assignment():
    result = read_experiment_values(None, [*RUN_A_TYPE_OVERRIDES, "+runs.a.x=1", "++runs.a.y=2"])
    assert result["runs"]["a"]["x"] == 1
    assert result["runs"]["a"]["y"] == 2


def test_tilde_deletes_a_run_and_a_key(tmp_path):
    experiment_path = _write_experiment(tmp_path, f"runs:\n  a:{RUN_TYPES_YAML}\n    num_rebuilds: 2\n  b: {{}}\n")

    result = read_experiment_values(experiment_path, ["~runs.b", "~runs.a.num_rebuilds"])

    assert result["runs"] == {
        "a": {"environment": {"type": "pick_and_place_maple_table"}, "policy": {"type": "zero_action"}}
    }


def test_tilde_with_a_value_is_rejected(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  a: {}\n  b: {}\n")

    with pytest.raises(AssertionError, match=r"'~runs\.b=x'.*Use ~runs\.b"):
        read_experiment_values(experiment_path, ["~runs.b=x"])


def test_comma_separated_value_is_rejected_with_a_list_hint():
    with pytest.raises(AssertionError, match=r"num_episodes_per_rebuild=\[value1,value2\] for a list"):
        read_experiment_values(None, ["runs.a.rollout_limit.num_episodes_per_rebuild=3,1"])


def test_mapping_value_replaces_the_old_mapping(tmp_path):
    experiment_path = _write_experiment(
        tmp_path, f"runs:\n  a:{RUN_TYPES_YAML}\n    rollout_limit:\n      num_episodes: 4\n"
    )

    result = read_experiment_values(experiment_path, ["runs.a.rollout_limit={num_steps:10}"])

    assert result["runs"]["a"]["rollout_limit"] == {"num_steps": 10}


def test_later_override_wins():
    result = read_experiment_values(None, [*RUN_A_TYPE_OVERRIDES, "runs.a.x=1", "runs.a.x=2"])
    assert result["runs"]["a"]["x"] == 2


def test_override_for_an_undeclared_run_lists_the_declared_runs(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  parallel_envs: {}\n  baseline: {}\n")

    with pytest.raises(
        AssertionError, match=r"not declared in the Experiment file: parallel_env\..*baseline, parallel_envs"
    ):
        read_experiment_values(experiment_path, ["runs.parallel_env.environment_builder.num_envs=8"])


def test_deleting_an_undeclared_run_lists_the_declared_runs(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  parallel_envs: {}\n  baseline: {}\n")

    with pytest.raises(
        AssertionError,
        match=r"Cannot delete 'runs\.parallel_env': the key does not exist\. "
        r"Keys under 'runs': baseline, parallel_envs",
    ):
        read_experiment_values(experiment_path, ["~runs.parallel_env"])


def test_deleting_below_a_missing_key_fails(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  baseline: {}\n")

    with pytest.raises(AssertionError, match=r"Cannot delete 'shared\.environment': the key does not exist\.$"):
        read_experiment_values(experiment_path, ["~shared.environment"])


def test_replacing_all_runs_cannot_add_a_run(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  baseline: {}\n")

    with pytest.raises(AssertionError, match=r"Runs not declared in the Experiment file: other"):
        read_experiment_values(experiment_path, ["runs={other:{}}"])


def test_file_without_runs_cannot_get_runs_from_overrides(tmp_path):
    experiment_path = _write_experiment(tmp_path, "name: empty\n")

    with pytest.raises(AssertionError, match=r"Overrides can only change the declared Runs: \(none\)"):
        read_experiment_values(experiment_path, ["runs.quick.environment.type=x"])


def test_file_without_runs_is_rejected(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs: {}\n")

    with pytest.raises(AssertionError, match=r"The Experiment must define at least one Run under 'runs' in '.*'"):
        read_experiment_values(experiment_path, [])


def test_command_line_without_runs_shows_a_run_override():
    with pytest.raises(
        AssertionError,
        match=r"Without an Experiment file, the overrides must define at least one Run, "
        r"for example runs\.<name>\.environment\.type=<type>",
    ):
        read_experiment_values(None, ["name=my_eval"])


def test_command_line_run_without_a_type_says_the_overrides_are_the_whole_experiment():
    # Without a file, this override does not change the smoke test Experiment, it defines a new Run.
    with pytest.raises(
        AssertionError,
        match=r"^Run 'smoke_test': missing 'environment\.type'\. "
        r"Set it, for example with runs\.smoke_test\.environment\.type=<type>\. "
        r"Without an Experiment file, the overrides are the whole Experiment\. "
        r"To change a Run of an Experiment file, pass the file\.$",
    ):
        read_experiment_values(None, ["runs.smoke_test.rollout_limit.num_steps=200"])


def test_file_run_without_a_type_names_the_run(tmp_path):
    experiment_path = _write_experiment(tmp_path, "runs:\n  a:\n    policy:\n      type: zero_action\n")

    with pytest.raises(
        AssertionError,
        match=r"^Run 'a': missing 'environment\.type'\. Set it, for example with runs\.a\.environment\.type=<type>$",
    ):
        read_experiment_values(experiment_path)


def test_type_that_is_not_a_string_names_the_run():
    with pytest.raises(AssertionError, match=r"^Run 'a': 'environment\.type' must be a string, got 1$"):
        read_experiment_values(None, ["runs.a.environment.type=1", "runs.a.policy.type=zero_action"])


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        pytest.param(
            [*RUN_A_TYPE_OVERRIDES, "shard.environment.light_intensity=5"],
            r"^Unknown Experiment fields: shard$",
            id="misspelled_top_level_field",
        ),
        pytest.param(
            ["runs.a.environment=pick_and_place_maple_table", "runs.a.policy.type=zero_action"],
            r"^Run 'a': 'environment' must be a mapping$",
            id="environment_that_is_not_a_mapping",
        ),
        pytest.param(
            [*RUN_A_TYPE_OVERRIDES, "runs.a.variations=[x]"],
            r"^Run 'a': 'variations' must be a mapping$",
            id="variations_that_are_not_a_mapping",
        ),
    ],
)
def test_override_with_a_misspelled_field_or_wrong_value_shape_is_rejected(overrides, expected_message):
    # Overrides create missing keys, so these checks are what catch a typo on the command line.
    with pytest.raises(AssertionError, match=expected_message):
        read_experiment_values(None, overrides)


def test_top_level_and_shared_overrides_are_applied():
    result = read_experiment_values(
        None,
        [*RUN_A_TYPE_OVERRIDES, "name=my_eval", "kpi_exporters=[]", "shared.environment.light_intensity=750.0"],
    )
    assert result["name"] == "my_eval"
    assert result["kpi_exporters"] == []
    assert result["runs"]["a"]["environment"]["light_intensity"] == 750.0


def test_experiment_without_a_file_uses_a_timestamped_name():
    experiment_name = read_experiment_values(None, RUN_A_TYPE_OVERRIDES)["name"]

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_experiment", experiment_name)


@pytest.mark.parametrize(("name_override", "expected_name"), [("name=2026", "2026"), ("name=1.5", "1.5")])
def test_numeric_experiment_name_becomes_a_string(name_override, expected_name):
    assert read_experiment_values(None, [*RUN_A_TYPE_OVERRIDES, name_override])["name"] == expected_name


@pytest.mark.parametrize("name_override", ["name=null", "name=true", "name=[a,b]"])
def test_experiment_name_that_is_not_a_string_or_number_is_rejected(name_override):
    with pytest.raises(AssertionError, match="Experiment 'name' must be a string"):
        read_experiment_values(None, [*RUN_A_TYPE_OVERRIDES, name_override])


def test_run_with_a_different_environment_type_does_not_inherit_shared_environment():
    runs = read_experiment_values(
        None,
        [
            "shared.environment={type:pick_and_place_maple_table,light_intensity:600.0}",
            "shared.policy.type=zero_action",
            "runs.a={}",
            "runs.b.environment.type=kitchen_pick_and_place",
        ],
    )["runs"]

    assert runs["a"]["environment"] == {"type": "pick_and_place_maple_table", "light_intensity": 600.0}
    assert runs["b"]["environment"] == {"type": "kitchen_pick_and_place"}


def test_run_with_a_different_policy_type_does_not_inherit_shared_policy():
    runs = read_experiment_values(
        None,
        [
            "shared.environment.type=pick_and_place_maple_table",
            "shared.policy={type:pi0_remote,remote_host:x}",
            "runs.a={}",
            "runs.b.policy.type=zero_action",
        ],
    )["runs"]

    assert runs["a"]["policy"] == {"type": "pi0_remote", "remote_host": "x"}
    assert runs["b"]["policy"] == {"type": "zero_action"}


def test_run_with_the_same_type_keeps_inheriting_shared_fields():
    runs = read_experiment_values(
        None,
        [
            "shared.environment={type:pick_and_place_maple_table,light_intensity:600.0}",
            "shared.policy.type=zero_action",
            "runs.a.environment={type:pick_and_place_maple_table,hdr:x}",
        ],
    )["runs"]

    assert runs["a"]["environment"] == {
        "type": "pick_and_place_maple_table",
        "light_intensity": 600.0,
        "hdr": "x",
    }


def test_run_field_override_without_a_section_inherits_shared_type(tmp_path):
    experiment_path = _write_experiment(
        tmp_path,
        "shared:\n  environment:\n    type: pick_and_place_maple_table\n  policy:\n    type: zero_action\n"
        "runs:\n  first: {}\n",
    )

    runs = read_experiment_values(experiment_path, ["runs.first.environment.light_intensity=700.0"])["runs"]

    assert runs["first"]["environment"] == {"type": "pick_and_place_maple_table", "light_intensity": 700.0}


def test_shared_merge_error_names_the_experiment_file_and_the_run(tmp_path):
    experiment_path = _write_experiment(
        tmp_path,
        "shared:\n  environment:\n    type: pick_and_place_maple_table\n"
        "runs:\n  first:\n    environment: [not, a, mapping]\n",
    )

    with pytest.raises(
        ValueError,
        match=rf"Could not read Arena Experiment '{re.escape(str(experiment_path))}': "
        r"Could not merge shared values into Run 'first'",
    ):
        read_experiment_values(experiment_path)


@pytest.mark.parametrize(
    ("config_name", "expected_run_count"),
    [
        ("robolab_18_tasks_pi0_and_cosmos.yaml", 36),
        ("robolab_20_tasks_pi0_and_cosmos.yaml", 40),
    ],
)
def test_robolab_experiments_apply_shared_run_defaults(config_name, expected_run_count):
    config_path = ROBOLAB_EXPERIMENT_CONFIG_DIRECTORY / config_name

    default_run_values = read_experiment_values(config_path)["runs"]
    enabled_run_values = read_experiment_values(
        config_path,
        ["shared.variations.droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true"],
    )["runs"]

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


DOTTED_VARIATION_YAML = f"""runs:
  a:{RUN_TYPES_YAML}
    variations:
      light.hdr_image.enabled: true
"""

DOTTED_SHARED_AND_NESTED_RUN_VARIATION_YAML = f"""shared:
  variations:
    light.hdr_image.enabled: true
runs:
  a:{RUN_TYPES_YAML}
    variations:
      light:
        hdr_image:
          enabled: false
"""


@pytest.mark.parametrize(
    ("contents", "overrides"),
    [
        (DOTTED_VARIATION_YAML, ["runs.a.variations.light.hdr_image.enabled=false"]),
        (DOTTED_SHARED_AND_NESTED_RUN_VARIATION_YAML, []),
    ],
    ids=["override_of_a_dotted_file_key", "dotted_shared_key_and_nested_run_key"],
)
def test_variation_set_both_as_a_dotted_key_and_nested_is_rejected(tmp_path, contents, overrides):
    experiment_path = _write_experiment(tmp_path, contents)

    with pytest.raises(
        AssertionError,
        match=r"Run 'a': variation 'light\.hdr_image\.enabled' is written both as a dotted key and nested\. "
        r"Write it one way; an override can only change an entry that the file writes nested\.",
    ):
        read_experiment_values(experiment_path, overrides)


def test_variation_paths_next_to_a_dotted_key_are_kept_as_written(tmp_path):
    experiment_path = _write_experiment(tmp_path, DOTTED_VARIATION_YAML)

    result = read_experiment_values(
        experiment_path,
        [
            "runs.a.variations.light.hdr_image.hdr_names=[garage_robolab]",
            "runs.a.variations.droid_rel_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.01,-0.01,-0.01]",
        ],
    )

    assert result["runs"]["a"]["variations"] == {
        "light.hdr_image.enabled": True,
        "light": {"hdr_image": {"hdr_names": ["garage_robolab"]}},
        "droid_rel_joint_pos": {"camera_extrinsics_wrist_camera": {"sampler_cfg": {"low": [-0.01, -0.01, -0.01]}}},
    }


def test_override_error_names_the_experiment_file_once(tmp_path):
    experiment_path = _write_experiment(tmp_path, f"shared:\n  environment: [1]\nruns:\n  a:{RUN_TYPES_YAML}\n")

    # The shared environment is a list, so the override cannot set a key in it.
    with pytest.raises(ValueError) as error:
        read_experiment_values(experiment_path, ["shared.environment.enable_cameras=true"])

    assert str(error.value).count(str(experiment_path)) == 1


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("runs.a.policy@foo=1", r"'runs\.a\.policy@foo=1' names a package with @"),
        ("runs.a.tags=extend_list(y)", r"'runs\.a\.tags=extend_list\(y\)' uses extend_list"),
    ],
)
def test_package_and_extend_list_overrides_are_rejected(override, error):
    with pytest.raises(AssertionError, match=error):
        read_experiment_values(None, [*RUN_A_TYPE_OVERRIDES, override])
