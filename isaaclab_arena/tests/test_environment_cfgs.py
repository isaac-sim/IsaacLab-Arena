# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the compatibility boundary between legacy CLI arguments and typed configs.

Each registered environment still accepts an ``argparse.Namespace`` through
``get_env()``. These tests replace ``build()`` so they can inspect the configuration
created at that boundary without starting Isaac Sim. Environment construction itself
is covered by the all-environments smoke test.
"""

import argparse
from dataclasses import fields, is_dataclass

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment import (
    GalileoG1StaticPickAndPlaceEnvironment,
)
from isaaclab_arena_environments.gr1_put_and_close_door_environment import GR1PutAndCloseDoorEnvironment
from isaaclab_arena_environments.gr1_table_multi_object_no_collision_environment import (
    GR1TableMultiObjectNoCollisionEnvironment,
)
from isaaclab_arena_environments.lift_object_environment import LiftObjectEnvironment
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironment

# TODO(cvolk, 2026-07-03): Delete these compatibility tests with the legacy argparse
# environment adapters.


def _environment_cfg_type(environment_type: type[ExampleEnvironmentBase]) -> type[ArenaEnvironmentCfg]:
    """Return the concrete configuration used by the legacy argparse adapter."""
    environment_cfg_type = environment_type._legacy_argparse_cfg_type
    assert (
        environment_cfg_type is not None
    ), f"{environment_type.__name__} does not declare a typed environment configuration"
    return environment_cfg_type


def _parse_legacy_arguments(
    environment_type: type[ExampleEnvironmentBase],
    environment_args: list[str] | None = None,
) -> argparse.Namespace:
    """Parse one environment's legacy options with the shared CLI defaults it consumes."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    environment_type.add_cli_args(parser)
    legacy_arguments = parser.parse_args([] if environment_args is None else environment_args)

    # These options belong to the shared Arena/Isaac Lab parser rather than an
    # environment subparser, but some compatibility adapters consume them.
    legacy_arguments.enable_cameras = False
    legacy_arguments.mimic = False
    legacy_arguments.num_envs = 1
    return legacy_arguments


def _capture_typed_cfg(
    monkeypatch,
    environment_type: type[ExampleEnvironmentBase],
    legacy_arguments: argparse.Namespace,
) -> ArenaEnvironmentCfg:
    """Return the config passed from the legacy ``get_env()`` adapter to ``build()``."""
    captured = {}
    expected_environment = object()

    def fake_build(self, cfg):
        captured["cfg"] = cfg
        return expected_environment

    monkeypatch.setattr(environment_type, "build", fake_build)
    factory = object.__new__(environment_type)

    environment = factory.get_env(legacy_arguments)

    assert environment is expected_environment
    return captured["cfg"]


def test_every_registered_legacy_adapter_translates_cli_defaults_to_its_typed_cfg(monkeypatch):
    """Check every ``get_env(args_cli)`` adapter at its legacy default values."""
    ensure_environments_registered()

    for environment_name in sorted(EnvironmentRegistry().get_all_keys()):
        environment_type = EnvironmentRegistry().get_component_by_name(environment_name)
        cfg_type = _environment_cfg_type(environment_type)
        assert is_dataclass(cfg_type), f"{cfg_type.__name__} must be a dataclass"

        legacy_arguments = _parse_legacy_arguments(environment_type)
        cfg = _capture_typed_cfg(monkeypatch, environment_type, legacy_arguments)

        assert type(cfg) is cfg_type
        assert isinstance(cfg, ArenaEnvironmentCfg)
        assert cfg == cfg_type(), f"{environment_name} CLI defaults diverged from its typed config defaults"


def test_generated_cli_arguments_and_cfg_validation(monkeypatch):
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

    for environment_type, cli_args, expected_values in test_cases:
        legacy_arguments = _parse_legacy_arguments(environment_type, cli_args)
        cfg = _capture_typed_cfg(monkeypatch, environment_type, legacy_arguments)
        for field_name, expected_value in expected_values.items():
            assert getattr(cfg, field_name) == expected_value

    invalid_mode_args = _parse_legacy_arguments(GR1TableMultiObjectNoCollisionEnvironment, ["--mode", "unsupported"])
    with pytest.raises(AssertionError, match="Unsupported placement mode"):
        _capture_typed_cfg(monkeypatch, GR1TableMultiObjectNoCollisionEnvironment, invalid_mode_args)


def test_maple_teleop_device_remains_a_cli_only_option(monkeypatch):
    """Keep Maple's teleop selection out of the environment construction config."""
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--teleop_device", "spacemouse"],
    )

    cfg = _capture_typed_cfg(monkeypatch, PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert legacy_arguments.teleop_device == "spacemouse"
    assert "teleop_device" not in {cfg_field.name for cfg_field in fields(cfg)}
