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
from dataclasses import fields, is_dataclass

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena_environments.cli import (
    _environment_cfg_from_cli,
    _get_legacy_argparse_cfg_type,
    add_environment_cli_args,
    build_environment_from_cli,
    ensure_environments_registered,
)
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment import (
    GalileoG1StaticPickAndPlaceEnvironment,
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

# TODO(cvolk, 2026-07-03): Delete these compatibility tests with the legacy argparse
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

    for environment_name in sorted(EnvironmentRegistry().get_all_keys()):
        environment_factory_type = EnvironmentRegistry().get_component_by_name(environment_name)
        assert issubclass(environment_factory_type, ArenaEnvironmentFactory)

        environment_cfg_type = _get_legacy_argparse_cfg_type(environment_factory_type)
        assert is_dataclass(environment_cfg_type), f"{environment_cfg_type.__name__} must be a dataclass"

        legacy_arguments = _parse_legacy_arguments(environment_factory_type)
        environment_cfg = _environment_cfg_from_cli(environment_factory_type, legacy_arguments)

        assert type(environment_cfg) is environment_cfg_type
        assert isinstance(environment_cfg, ArenaEnvironmentCfg)
        assert (
            environment_cfg == environment_cfg_type()
        ), f"{environment_name} CLI defaults diverged from its typed config defaults"


def test_generated_cli_arguments_and_cfg_validation():
    """Keep list, boolean, and scalar parsing while configs validate domain values."""
    test_cases = [
        (
            GR1PutAndCloseDoorEnvironment,
            ["--object_set", "cracker_box", "mustard_bottle"],
            {"object_set": ["cracker_box", "mustard_bottle"]},
        ),
        (GR1PutAndCloseDoorEnvironment, ["--object_set"], {"object_set": []}),
        (GalileoG1StaticPickAndPlaceEnvironment, ["--no-lock_waist"], {"lock_waist": False}),
        (LiftObjectEnvironment, ["--rl_training_mode"], {"rl_training_mode": True}),
        (GR1TableMultiObjectNoCollisionEnvironment, ["--mode", "heterogeneous"], {"mode": "heterogeneous"}),
        (
            PickAndPlaceMapleTableEnvironment,
            ["--light_intensity", "750", "--additional_table_objects", "apple", "banana"],
            {"light_intensity": 750.0, "additional_table_objects": ["apple", "banana"]},
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


def test_build_environment_from_cli_keeps_external_legacy_path():
    """Dispatch an external compatibility factory through its own argparse methods."""
    expected_environment = object()

    class LegacyExternalEnvironment(ExampleEnvironmentBase):
        name = "legacy_external"

        def __init__(self):
            pass

        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--selection", default="default")

        def get_env(self, args_cli):
            assert args_cli.selection == "custom"
            return expected_environment

    parser = argparse.ArgumentParser(exit_on_error=False)
    add_environment_cli_args(parser, LegacyExternalEnvironment)
    legacy_arguments = parser.parse_args(["--selection", "custom"])

    environment = build_environment_from_cli(LegacyExternalEnvironment, legacy_arguments)

    assert environment is expected_environment


def test_maple_teleop_device_remains_a_cli_only_option():
    """Keep Maple's teleop selection out of the environment construction config."""
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--teleop_device", "spacemouse"],
    )
    environment_cfg = _environment_cfg_from_cli(PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert legacy_arguments.teleop_device == "spacemouse"
    assert "teleop_device" not in {config_field.name for config_field in fields(environment_cfg)}
