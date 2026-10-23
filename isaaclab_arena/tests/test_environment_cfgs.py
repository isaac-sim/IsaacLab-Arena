# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the temporary argparse adapter for typed environment factories.

The production CLI frontend generates flags from each environment config dataclass
and converts the resulting Namespace back into that same config type. These tests
exercise that boundary without constructing an Isaac Sim environment.
"""

import argparse
import sys
import types
from dataclasses import dataclass, fields, is_dataclass

import pytest

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena_environments.cli import (
    _environment_cfg_from_cli,
    add_environment_cli_args,
    build_environment_from_cli,
    ensure_environments_registered,
    get_isaaclab_arena_environments_cli_parser,
    import_external_environment,
)
from isaaclab_arena_environments.gr1_put_and_close_door_environment import GR1PutAndCloseDoorEnvironment
from isaaclab_arena_environments.gr1_table_multi_object_no_collision_environment import (
    GR1TableMultiObjectNoCollisionEnvironment,
)
from isaaclab_arena_environments.lift_object_environment import LiftObjectEnvironment
from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)

# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete these compatibility tests with the legacy argparse
# environment adapter.


def _parse_legacy_arguments(
    environment_factory_type: type[ArenaEnvironmentFactory],
    environment_args: list[str] | None = None,
) -> argparse.Namespace:
    """Parse one environment's generated flags plus its shared CLI values."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_environment_cli_args(parser, environment_factory_type)
    legacy_arguments = parser.parse_args([] if environment_args is None else environment_args)

    # These values normally come from the shared Arena parser or a specialized runner.
    legacy_arguments.enable_cameras = False
    legacy_arguments.mimic = False
    legacy_arguments.num_envs = 1
    return legacy_arguments


def test_core_environment_factory_does_not_expose_argparse_methods():
    """Keep CLI compatibility outside the core factory contract."""
    assert not hasattr(ArenaEnvironmentFactory, "add_cli_args")
    assert not hasattr(ArenaEnvironmentFactory, "get_env")


def test_every_registered_cli_adapter_uses_its_typed_cfg_defaults():
    """Generate every environment's default Namespace and recover its config."""
    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()

    for environment_name in sorted(environment_registry.get_all_keys()):
        environment_factory_type = environment_registry.get_component_by_name(environment_name)
        assert issubclass(environment_factory_type, ArenaEnvironmentFactory)

        environment_cfg_type = environment_registry.get_environment_cfg_type(environment_factory_type)
        assert is_dataclass(environment_cfg_type), f"{environment_cfg_type.__name__} must be a dataclass"

        legacy_arguments = _parse_legacy_arguments(environment_factory_type)
        environment_cfg = _environment_cfg_from_cli(environment_factory_type, legacy_arguments)

        assert type(environment_cfg) is environment_cfg_type
        assert isinstance(environment_cfg, ArenaEnvironmentCfg)
        assert environment_registry.get_factory_type_for_cfg(environment_cfg) is environment_factory_type
        assert (
            environment_cfg == environment_cfg_type()
        ), f"{environment_name} CLI defaults diverged from its typed config defaults"


def test_generated_cli_arguments_and_cfg_validation():
    """Keep list, boolean, and scalar parsing while configs validate domain values."""
    assert PickAndPlaceMapleTableEnvironmentCfg().episode_length_s == 70.0

    test_cases = [
        (
            GR1PutAndCloseDoorEnvironment,
            ["--object_set", "cracker_box", "mustard_bottle"],
            {"object_set": ["cracker_box", "mustard_bottle"]},
        ),
        (GR1PutAndCloseDoorEnvironment, ["--object_set"], {"object_set": []}),
        (LiftObjectEnvironment, ["--rl_training_mode"], {"rl_training_mode": True}),
        (GR1TableMultiObjectNoCollisionEnvironment, ["--mode", "heterogeneous"], {"mode": "heterogeneous"}),
        (
            PickAndPlaceMapleTableEnvironment,
            [
                "--light_intensity",
                "750",
                "--additional_table_objects",
                "apple",
                "banana",
                "--episode_length_s",
                "1.5",
            ],
            {
                "light_intensity": 750.0,
                "additional_table_objects": ["apple", "banana"],
                "episode_length_s": 1.5,
            },
        ),
    ]

    for environment_factory_type, cli_args, expected_values in test_cases:
        legacy_arguments = _parse_legacy_arguments(environment_factory_type, cli_args)
        environment_cfg = _environment_cfg_from_cli(environment_factory_type, legacy_arguments)
        for field_name, expected_value in expected_values.items():
            assert getattr(environment_cfg, field_name) == expected_value

    invalid_mode_args = _parse_legacy_arguments(GR1TableMultiObjectNoCollisionEnvironment, ["--mode", "unsupported"])
    with pytest.raises(AssertionError, match="Unsupported placement mode"):
        _environment_cfg_from_cli(GR1TableMultiObjectNoCollisionEnvironment, invalid_mode_args)

    invalid_episode_length_args = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--episode_length_s", "0"],
    )
    with pytest.raises(AssertionError, match="episode_length_s must be greater than zero"):
        _environment_cfg_from_cli(PickAndPlaceMapleTableEnvironment, invalid_episode_length_args)


def test_build_environment_from_cli_calls_typed_build(monkeypatch):
    """Dispatch a first-party CLI request through ``build(cfg)``."""
    captured = {}
    expected_environment = object()

    def fake_build(self, environment_cfg):
        captured["environment_cfg"] = environment_cfg
        return expected_environment

    monkeypatch.setattr(PickAndPlaceMapleTableEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(PickAndPlaceMapleTableEnvironment, "build", fake_build)
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--light_intensity", "750"],
    )

    environment = build_environment_from_cli(PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert environment is expected_environment
    assert captured["environment_cfg"] == PickAndPlaceMapleTableEnvironmentCfg(light_intensity=750.0)


def _add_importable_test_module(monkeypatch, module_name: str, **module_attributes) -> None:
    """Make a throwaway module importable by name for the duration of one test."""
    test_module = types.ModuleType(module_name)
    for attribute_name, attribute_value in module_attributes.items():
        setattr(test_module, attribute_name, attribute_value)
    monkeypatch.setitem(sys.modules, module_name, test_module)


def test_external_environment_flag_adds_the_registered_factory_subcommand(monkeypatch):
    """Select an external typed factory through the environment subcommand parser used by runner scripts."""
    from isaaclab_arena_examples.external_environments.basic import (
        ExternalFrankaTableEnvironment,
        ExternalFrankaTableEnvironmentCfg,
    )

    environment_class_path = "isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment"
    cli_args = ["--external_environment_class_path", environment_class_path, "franka_table", "--object", "mug"]
    monkeypatch.setattr(sys, "argv", ["environment_runner.py", *cli_args])

    args_cli = get_isaaclab_arena_environments_cli_parser().parse_args(cli_args)
    # Graph-YAML Runs rebuild this parser, which imports the same factory again.
    get_isaaclab_arena_environments_cli_parser()

    assert args_cli.example_environment == "franka_table"
    assert EnvironmentRegistry().get_component_by_name("franka_table") is ExternalFrankaTableEnvironment
    environment_cfg = _environment_cfg_from_cli(ExternalFrankaTableEnvironment, args_cli)
    assert environment_cfg == ExternalFrankaTableEnvironmentCfg(object="mug")


def test_external_environment_flag_rejects_an_unregistered_factory(monkeypatch):
    """Require the factory's @register_environment decorator instead of registering it without a config."""

    @dataclass
    class UnregisteredEnvironmentCfg(ArenaEnvironmentCfg):
        pass

    class UnregisteredEnvironment(ArenaEnvironmentFactory[UnregisteredEnvironmentCfg]):
        name = "unregistered_external_environment"

    _add_importable_test_module(
        monkeypatch, "arena_test_unregistered_environment", UnregisteredEnvironment=UnregisteredEnvironment
    )

    with pytest.raises(AssertionError, match="Decorate it with @register_environment"):
        import_external_environment("arena_test_unregistered_environment:UnregisteredEnvironment")
    assert not EnvironmentRegistry().is_registered("unregistered_external_environment")


def test_external_environment_flag_rejects_a_built_in_name(monkeypatch):
    """Reject an external factory whose name the decorator left with a built-in environment."""
    ensure_environments_registered()

    @dataclass
    class DuplicateNameEnvironmentCfg(ArenaEnvironmentCfg):
        pass

    # The decorator keeps the built-in kitchen_pick_and_place factory and skips this one.
    @register_environment
    class DuplicateNameEnvironment(ArenaEnvironmentFactory[DuplicateNameEnvironmentCfg]):
        name = "kitchen_pick_and_place"

    _add_importable_test_module(
        monkeypatch, "arena_test_duplicate_name_environment", DuplicateNameEnvironment=DuplicateNameEnvironment
    )

    with pytest.raises(AssertionError, match="'kitchen_pick_and_place', which is already registered"):
        import_external_environment("arena_test_duplicate_name_environment:DuplicateNameEnvironment")


def test_maple_teleop_device_remains_a_cli_only_option():
    """Keep Maple's teleop selection out of the environment construction config."""
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--teleop_device", "spacemouse"],
    )
    environment_cfg = _environment_cfg_from_cli(PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert legacy_arguments.teleop_device == "spacemouse"
    assert "teleop_device" not in {config_field.name for config_field in fields(environment_cfg)}
