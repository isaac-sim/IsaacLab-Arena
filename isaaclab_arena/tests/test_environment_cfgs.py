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
from typing import get_args, get_origin

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironment


def _environment_cfg_type(environment_type: type[ExampleEnvironmentBase]) -> type[ArenaEnvironmentCfg]:
    """Return the concrete configuration type declared by an environment factory."""
    for base in environment_type.__orig_bases__:
        if get_origin(base) is ExampleEnvironmentBase:
            (cfg_type,) = get_args(base)
            return cfg_type
    raise AssertionError(f"{environment_type.__name__} does not declare a typed environment configuration")


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
        expected_cfg = cfg_type(**{
            cfg_field.name: getattr(legacy_arguments, cfg_field.name)
            for cfg_field in fields(cfg)
            if hasattr(legacy_arguments, cfg_field.name)
        })
        assert cfg == expected_cfg, f"{environment_name} did not retain its legacy defaults"


def test_maple_teleop_device_remains_a_cli_only_option(monkeypatch):
    """Keep Maple's teleop selection out of the environment construction config."""
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--teleop_device", "spacemouse"],
    )

    cfg = _capture_typed_cfg(monkeypatch, PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert legacy_arguments.teleop_device == "spacemouse"
    assert "teleop_device" not in {cfg_field.name for cfg_field in fields(cfg)}
