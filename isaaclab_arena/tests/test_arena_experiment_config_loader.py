# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test loading Arena Experiments at the evaluation boundary."""

import importlib
import re
import sys
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.evaluation import arena_experiment_config_loader, experiment_runner
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import (
    load_arena_experiment_from_config_file,
    validate_experiment_config_path,
)
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.experiment_runner import _assert_camera_support_enabled
from isaaclab_arena.evaluation.experiment_runner_cli import parse_experiment_runner_args
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.test_environment_cfgs import _write_external_factory_module, _write_external_module
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.kitchen_pick_and_place_environment import KitchenPickAndPlaceEnvironmentCfg
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

GETTING_STARTED_YAML_PATH = (
    Path(TestConstants.arena_environments_dir) / "experiment_configs" / "getting_started_experiment.yaml"
)
CAMERA_SENSITIVITY_YAML_PATH = (
    Path(TestConstants.arena_environments_dir)
    / "experiment_configs"
    / "droid_pnp_camera_sensitivity_openpi_experiment.yaml"
)


def _experiment_cfg(enable_cameras: bool = False) -> ArenaExperimentCfg:
    run_cfg = ArenaRunCfg(
        name="baseline",
        environment=PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=enable_cameras),
        policy=ZeroActionPolicyCfg(),
    )
    return ArenaExperimentCfg(runs={run_cfg.name: run_cfg}, name="experiment")


def test_load_typed_yaml_experiment_applies_overrides_and_device(monkeypatch):
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_resolve_policy_cfg_type_from_name_or_class_path",
        lambda policy_name_or_class_path: {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path],
    )

    experiment_cfg = load_arena_experiment_from_config_file(
        GETTING_STARTED_YAML_PATH,
        device="cuda:1",
        overrides=["runs.baseline.environment.light_intensity=750.0"],
    )
    runs = experiment_cfg.runs

    assert isinstance(experiment_cfg, ArenaExperimentCfg)
    assert list(runs) == [
        "baseline",
        "swap_objects",
        "change_background_hdr",
        "parallel_envs",
    ]
    assert isinstance(runs["baseline"].environment, PickAndPlaceMapleTableEnvironmentCfg)
    assert runs["baseline"].environment.light_intensity == 750.0
    assert all(run.policy == ZeroActionPolicyCfg() for run in runs.values())
    assert all(run.environment_builder.device == "cuda:1" for run in runs.values())


def _patch_typed_loader_registries(monkeypatch):
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_resolve_policy_cfg_type_from_name_or_class_path",
        lambda policy_name_or_class_path: {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path],
    )


def test_name_defaults_to_the_yaml_stem(monkeypatch):
    _patch_typed_loader_registries(monkeypatch)

    experiment_cfg = load_arena_experiment_from_config_file(GETTING_STARTED_YAML_PATH, device="cuda:0")

    assert experiment_cfg.name == "getting_started_experiment"


def test_name_in_the_file_wins_over_the_stem(tmp_path, monkeypatch):
    _patch_typed_loader_registries(monkeypatch)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "name": "named-in-file",
            "runs": {
                "baseline": {
                    "environment": {"type": "pick_and_place_maple_table"},
                    "policy": {"type": "zero_action"},
                }
            },
        }),
        encoding="utf-8",
    )

    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:0")

    assert experiment_cfg.name == "named-in-file"


def test_name_override_wins(monkeypatch):
    _patch_typed_loader_registries(monkeypatch)

    experiment_cfg = load_arena_experiment_from_config_file(
        GETTING_STARTED_YAML_PATH, device="cuda:0", overrides=["name=my-benchmark"]
    )

    assert experiment_cfg.name == "my-benchmark"


@pytest.mark.parametrize("name", [None, "", "two\nlines"])
def test_experiment_cfg_requires_a_non_empty_single_line_name(name):
    run_cfg = _experiment_cfg().runs["baseline"]

    with pytest.raises(AssertionError, match="name"):
        ArenaExperimentCfg(runs={"baseline": run_cfg}, name=name)


def test_load_typed_yaml_experiment_with_graph_spec_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(arena_experiment_config_loader, "_registered_environment_cfg_types", lambda: {})
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_resolve_policy_cfg_type_from_name_or_class_path",
        lambda policy_name_or_class_path: {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path],
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
runs:
  graph_run:
    environment:
      type: robolab/tasks/banana_in_bowl.yaml
      enable_cameras: true
      pick_up_object: banana
    policy:
      type: zero_action
    environment_builder:
      num_envs: 2
      language_instruction: Pick up the banana.
    rollout_limit:
      num_episodes: 4
""",
        encoding="utf-8",
    )

    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:1")
    run = experiment_cfg.runs["graph_run"]

    assert isinstance(run.environment, LegacyGraphEnvironmentCfg)
    # Only environment values are stored (and later become CLI tokens at execution); the
    # environment_builder section stays typed and reaches the argparse path directly.
    assert run.environment.env_spec_path == "robolab/tasks/banana_in_bowl.yaml"
    assert run.environment.per_run_overrides == {"enable_cameras": True, "pick_up_object": "banana"}
    assert run.environment_builder.num_envs == 2
    assert run.environment_builder.device == "cuda:1"
    assert run.environment_builder.language_instruction == "Pick up the banana."
    assert run.rollout_limit.num_episodes == 4


def test_typed_experiment_ignores_unconfigured_external_environment(monkeypatch):
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
    from isaaclab_arena_environments.cli import ensure_environments_registered

    class EnvironmentWithoutConfig(ArenaEnvironmentFactory[ArenaEnvironmentCfg]):
        name = "external_without_config"

    ensure_environments_registered()
    registry = EnvironmentRegistry()
    monkeypatch.setattr(registry, "_components", registry._components.copy())
    # Plain register() records no config, unlike @register_environment.
    registry.register(EnvironmentWithoutConfig, EnvironmentWithoutConfig.name)

    experiment = load_arena_experiment_from_config_file(GETTING_STARTED_YAML_PATH, device="cuda:0")
    assert all(isinstance(run.environment, PickAndPlaceMapleTableEnvironmentCfg) for run in experiment.runs.values())

    with pytest.raises(ValueError, match="Run 'baseline' could not resolve environment type 'external_without_config'"):
        load_arena_experiment_from_config_file(
            GETTING_STARTED_YAML_PATH,
            device="cuda:0",
            overrides=["shared.environment.type=external_without_config"],
        )

    # Typed loading keeps registered aliases.
    factory = registry.get_component_by_name("pick_and_place_maple_table")
    registry.register(factory, "maple_alias")
    configured = registry.get_environment_cfg_types()
    assert configured["maple_alias"] is configured["pick_and_place_maple_table"]
    assert "external_without_config" not in configured
    experiment = load_arena_experiment_from_config_file(
        GETTING_STARTED_YAML_PATH,
        device="cuda:0",
        overrides=["shared.environment.type=maple_alias"],
    )
    assert all(isinstance(run.environment, PickAndPlaceMapleTableEnvironmentCfg) for run in experiment.runs.values())


def test_graph_spec_environment_serializes_to_reloadable_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(arena_experiment_config_loader, "_registered_environment_cfg_types", lambda: {})
    monkeypatch.setattr(
        arena_experiment_config_loader,
        "_resolve_policy_cfg_type_from_name_or_class_path",
        lambda policy_name_or_class_path: {"zero_action": ZeroActionPolicyCfg}[policy_name_or_class_path],
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
runs:
  graph_run:
    environment:
      type: robolab/tasks/banana_in_bowl.yaml
      enable_cameras: true
      pick_up_object: banana
    policy:
      type: zero_action
    environment_builder:
      num_envs: 2
    rollout_limit:
      num_episodes: 4
""",
        encoding="utf-8",
    )
    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:1")

    serialized_experiment = serialize_arena_experiment_to_yaml(experiment_cfg)
    serialized_values = yaml.safe_load(serialized_experiment)
    serialized_environment = serialized_values["runs"]["graph_run"]["environment"]
    assert serialized_environment == {
        "type": "robolab/tasks/banana_in_bowl.yaml",
        "enable_cameras": True,
        "pick_up_object": "banana",
    }

    serialized_path = tmp_path / "serialized_experiment.yaml"
    serialized_path.write_text(serialized_experiment, encoding="utf-8")
    assert load_arena_experiment_from_config_file(serialized_path, device="cuda:1") == experiment_cfg


def test_typed_graph_camera_run_requires_prelaunch_camera_flag():
    run_cfg = ArenaRunCfg(
        name="graph_run",
        environment=LegacyGraphEnvironmentCfg(
            env_spec_path="robolab/tasks/banana_in_bowl.yaml",
            enable_cameras=True,
        ),
        policy=ZeroActionPolicyCfg(),
    )
    experiment_cfg = ArenaExperimentCfg(runs={run_cfg.name: run_cfg}, name="experiment")

    with pytest.raises(AssertionError, match="enable environment cameras"):
        _assert_camera_support_enabled(experiment_cfg, enable_cameras=False)

    _assert_camera_support_enabled(experiment_cfg, enable_cameras=True)


def test_policy_config_type_resolves_from_dotted_class_path():
    policy_cfg_type = arena_experiment_config_loader._resolve_policy_cfg_type_from_name_or_class_path(
        "isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy"
    )

    assert policy_cfg_type is ZeroActionPolicyCfg


@pytest.fixture
def isolated_environment_registry(monkeypatch):
    """Remove the environments that modules imported during a test register once the test ends."""
    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()
    monkeypatch.setattr(environment_registry, "_components", environment_registry._components.copy())
    monkeypatch.setattr(
        environment_registry,
        "_factory_types_by_cfg_type",
        environment_registry._factory_types_by_cfg_type.copy(),
    )


def _write_experiment_selecting_environment_type(tmp_path: Path, environment_type: str) -> Path:
    """Write an Experiment whose Run 'valid' selects a built-in environment and Run 'selected' environment_type."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "runs": {
                "valid": {"environment": {"type": "pick_and_place_maple_table"}, "policy": {"type": "zero_action"}},
                "selected": {"environment": {"type": environment_type}, "policy": {"type": "zero_action"}},
            }
        }),
        encoding="utf-8",
    )
    return config_path


def _assert_selected_run_cannot_resolve_environment_type(config_path: Path, environment_type: str, error: str):
    """Check that loading fails with an error naming Run 'selected', its environment type, and the cause."""
    expected_message = (
        f"Run 'selected' could not resolve environment type '{re.escape(environment_type)}': .*{re.escape(error)}"
    )
    with pytest.raises(ValueError, match=expected_message):
        load_arena_experiment_from_config_file(config_path, device="cuda:0")


def _write_experiment_selecting_external_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, environment_name: str
) -> tuple[Path, str]:
    """Write an external factory module and an Experiment whose Run 'external' selects it by dotted class path.

    Returns:
        The Experiment path and the factory's dotted class path.
    """
    flag_class_path = _write_external_factory_module(tmp_path, monkeypatch, module_name, environment_name)
    # The flag joins module and class with ":"; Experiments use a dot.
    environment_type = flag_class_path.replace(":", ".")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "runs": {
                "external": {
                    "environment": {"type": environment_type, "object": "mug"},
                    "policy": {"type": "zero_action"},
                }
            }
        }),
        encoding="utf-8",
    )
    return config_path, environment_type


def test_environment_type_selects_an_external_factory_by_dotted_class_path(
    tmp_path, monkeypatch, isolated_environment_registry
):
    """Import a factory that only the Experiment names, and compose its config from the Run values."""
    config_path, _ = _write_experiment_selecting_external_factory(
        tmp_path, monkeypatch, "experiment_selected_environment_module", "experiment_selected_environment"
    )

    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:0")

    external_module = sys.modules["experiment_selected_environment_module"]
    assert experiment_cfg.runs["external"].environment == external_module.ExternalEnvironmentCfg(object="mug")
    assert EnvironmentRegistry().get_component_by_name("experiment_selected_environment") is (
        external_module.ExternalEnvironment
    )


@pytest.mark.parametrize(
    ("environment_type", "error"),
    [
        ("missing_environment", "is not a registered environment name or a dotted class path"),
        ("missing_package.environments.MissingEnvironment", "No module named 'missing_package'"),
        (
            "isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment",
            "uses the command-line form 'module_path:ClassName'. Experiments use the dotted form",
        ),
    ],
)
def test_unresolvable_environment_type_names_the_run(tmp_path, environment_type, error):
    config_path = _write_experiment_selecting_environment_type(tmp_path, environment_type)

    _assert_selected_run_cannot_resolve_environment_type(config_path, environment_type, error)


def test_environment_type_rejects_an_unregistered_factory(tmp_path, monkeypatch, isolated_environment_registry):
    flag_class_path = _write_external_factory_module(
        tmp_path,
        monkeypatch,
        "experiment_unregistered_environment_module",
        "experiment_unregistered_environment",
        decorate=False,
    )
    environment_type = flag_class_path.replace(":", ".")
    config_path = _write_experiment_selecting_environment_type(tmp_path, environment_type)

    _assert_selected_run_cannot_resolve_environment_type(
        config_path, environment_type, "is not registered. Decorate it with @register_environment."
    )


def test_environment_type_names_the_run_when_its_module_fails_to_import(tmp_path, monkeypatch):
    """Name the Run when the factory module exists but raises an error that is not an ImportError."""
    _write_external_module(tmp_path, monkeypatch, "environment_module_failing_at_import", "undefined_helper()\n")
    environment_type = "environment_module_failing_at_import.ExternalEnvironment"
    config_path = _write_experiment_selecting_environment_type(tmp_path, environment_type)

    _assert_selected_run_cannot_resolve_environment_type(
        config_path, environment_type, "name 'undefined_helper' is not defined"
    )


def test_environment_type_rejects_a_factory_registered_without_its_config(
    tmp_path, monkeypatch, isolated_environment_registry
):
    _write_external_module(
        tmp_path,
        monkeypatch,
        "environment_registered_without_config_module",
        """
        from isaaclab_arena.assets.registries import EnvironmentRegistry
        from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


        class EnvironmentWithoutConfig(ArenaEnvironmentFactory[ArenaEnvironmentCfg]):
            name = "environment_registered_without_config"


        # Plain register() records no config, unlike @register_environment.
        EnvironmentRegistry().register(EnvironmentWithoutConfig, EnvironmentWithoutConfig.name)
        """,
    )
    environment_type = "environment_registered_without_config_module.EnvironmentWithoutConfig"
    config_path = _write_experiment_selecting_environment_type(tmp_path, environment_type)

    _assert_selected_run_cannot_resolve_environment_type(
        config_path, environment_type, "is registered without its config. Register it with @register_environment"
    )


def test_environment_type_rejects_a_factory_that_reuses_a_built_in_name(
    tmp_path, monkeypatch, isolated_environment_registry
):
    flag_class_path = _write_external_factory_module(
        tmp_path, monkeypatch, "experiment_taken_name_environment_module", "kitchen_pick_and_place"
    )
    environment_type = flag_class_path.replace(":", ".")
    config_path = _write_experiment_selecting_environment_type(tmp_path, environment_type)

    _assert_selected_run_cannot_resolve_environment_type(
        config_path,
        environment_type,
        "uses the name 'kitchen_pick_and_place', which is taken by"
        " isaaclab_arena_environments.kitchen_pick_and_place_environment.KitchenPickAndPlaceEnvironment",
    )
    # The built-in environment keeps its name.
    built_in_config_path = _write_experiment_selecting_environment_type(tmp_path, "kitchen_pick_and_place")
    experiment_cfg = load_arena_experiment_from_config_file(built_in_config_path, device="cuda:0")
    assert isinstance(experiment_cfg.runs["selected"].environment, KitchenPickAndPlaceEnvironmentCfg)


def test_external_environment_serializes_to_its_class_path(tmp_path, monkeypatch, isolated_environment_registry):
    """Write the class path, which a fresh process can load before anything registers the factory's name."""
    config_path, environment_type = _write_experiment_selecting_external_factory(
        tmp_path, monkeypatch, "serialized_external_environment_module", "serialized_external_environment"
    )
    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:0")

    serialized_experiment = serialize_arena_experiment_to_yaml(experiment_cfg)

    serialized_environment = yaml.safe_load(serialized_experiment)["runs"]["external"]["environment"]
    assert serialized_environment["type"] == environment_type
    assert serialized_environment["object"] == "mug"
    serialized_path = tmp_path / "serialized_experiment.yaml"
    serialized_path.write_text(serialized_experiment, encoding="utf-8")
    assert load_arena_experiment_from_config_file(serialized_path, device="cuda:0") == experiment_cfg


def test_serializer_rejects_a_nested_environment_factory(tmp_path, monkeypatch, isolated_environment_registry):
    """Stop instead of writing a class path that the loader cannot import."""
    _write_external_module(
        tmp_path,
        monkeypatch,
        "nested_environment_factory_module",
        """
        from dataclasses import dataclass

        from isaaclab_arena.assets.register import register_environment
        from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


        @dataclass
        class NestedEnvironmentCfg(ArenaEnvironmentCfg):
            pass


        class EnvironmentGroup:

            @register_environment
            class NestedEnvironment(ArenaEnvironmentFactory[NestedEnvironmentCfg]):
                name = "nested_environment_factory"

                def build(self, cfg):
                    raise NotImplementedError
        """,
    )
    nested_module = importlib.import_module("nested_environment_factory_module")
    run_cfg = ArenaRunCfg(name="nested", environment=nested_module.NestedEnvironmentCfg(), policy=ZeroActionPolicyCfg())
    experiment_cfg = ArenaExperimentCfg(runs={run_cfg.name: run_cfg}, name="experiment")

    with pytest.raises(
        AssertionError,
        match=(
            r"nested_environment_factory_module\.EnvironmentGroup\.NestedEnvironment cannot be imported as"
            r" 'nested_environment_factory_module\.NestedEnvironment'"
        ),
    ):
        serialize_arena_experiment_to_yaml(experiment_cfg)


def test_default_experiment_is_a_short_zero_action_smoke_test(monkeypatch):
    """Running experiment_runner.py without --experiment_config runs one short zero-action Run."""
    from isaaclab_arena_environments.cube_goal_pose_environment import CubeGoalPoseEnvironmentCfg

    # The default path is relative to the repository root, where the documented command runs.
    monkeypatch.chdir(TestConstants.repo_root)
    args_cli, _ = parse_experiment_runner_args([])

    experiment_cfg = load_arena_experiment_from_config_file(args_cli.experiment_config, device="cuda:0")

    assert list(experiment_cfg.runs) == ["smoke_test"]
    smoke_test_run = experiment_cfg.runs["smoke_test"]
    assert isinstance(smoke_test_run.environment, CubeGoalPoseEnvironmentCfg)
    assert smoke_test_run.policy == ZeroActionPolicyCfg()
    assert smoke_test_run.rollout_limit.num_steps == 20


def test_experiment_runner_loads_typed_experiment_after_simulation_starts(monkeypatch):
    simulation_is_running = False

    class _SimulationAppContext:
        def __init__(self, _args_cli):
            pass

        def __enter__(self):
            nonlocal simulation_is_running
            simulation_is_running = True

        def __exit__(self, _exception_type, _exception, _traceback):
            nonlocal simulation_is_running
            simulation_is_running = False

    def load_experiment_after_startup(*_args, **_kwargs):
        assert simulation_is_running
        return _experiment_cfg()

    monkeypatch.setattr(experiment_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(experiment_runner, "load_arena_experiment_from_config_file", load_experiment_after_startup)
    monkeypatch.setattr(experiment_runner, "list_variations", lambda _experiment: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--list_variations",
        ],
    )

    experiment_runner.main()


def test_camera_support_invariant_holds_for_composed_runs():
    experiment_cfg = _experiment_cfg(enable_cameras=True)

    with pytest.raises(AssertionError, match="AppLauncher started without camera support"):
        _assert_camera_support_enabled(experiment_cfg, enable_cameras=False)

    _assert_camera_support_enabled(experiment_cfg, enable_cameras=True)
    _assert_camera_support_enabled(_experiment_cfg(enable_cameras=False), enable_cameras=False)


@pytest.mark.parametrize(
    ("experiment_config_path", "expected_enable_cameras"),
    [(CAMERA_SENSITIVITY_YAML_PATH, True), (GETTING_STARTED_YAML_PATH, False)],
)
def test_typed_experiment_enables_camera_support_from_its_config(
    monkeypatch, experiment_config_path, expected_enable_cameras
):
    """A typed Experiment that declares cameras must not also need --enable_cameras on the CLI."""
    launched_with_cameras = None

    class _SimulationAppContext:
        def __init__(self, args_cli):
            nonlocal launched_with_cameras
            launched_with_cameras = args_cli.enable_cameras

        def __enter__(self):
            pass

        def __exit__(self, _exception_type, _exception, _traceback):
            pass

    monkeypatch.setattr(experiment_runner, "SimulationAppContext", _SimulationAppContext)
    monkeypatch.setattr(
        experiment_runner,
        "load_arena_experiment_from_config_file",
        lambda *_args, **_kwargs: _experiment_cfg(enable_cameras=expected_enable_cameras),
    )
    monkeypatch.setattr(experiment_runner, "list_variations", lambda _experiment: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(experiment_config_path),
            "--list_variations",
        ],
    )

    experiment_runner.main()

    assert launched_with_cameras is expected_enable_cameras


def test_experiment_runner_rejects_nonempty_exact_output_before_starting_simulation(monkeypatch, tmp_path):
    exact_experiment_output_directory = tmp_path / "existing-experiment-output"
    exact_experiment_output_directory.mkdir()
    (exact_experiment_output_directory / "existing-result.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_runner.py",
            "--experiment_config",
            str(GETTING_STARTED_YAML_PATH),
            "--experiment_output_directory",
            str(exact_experiment_output_directory),
        ],
    )

    with pytest.raises(AssertionError, match="is not empty.*--output_base_dir"):
        experiment_runner.main()


@pytest.mark.parametrize("file_name", ["experiment.toml", "experiment.json"])
def test_unknown_experiment_config_file_type_is_rejected(tmp_path, file_name):
    config_path = tmp_path / file_name
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="must be a YAML file"):
        validate_experiment_config_path(config_path)
