# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for registered environments' typed configurations."""

import argparse
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


def _environment_cfg_type(environment_type: type[ExampleEnvironmentBase]) -> type[ArenaEnvironmentCfg]:
    """Return the concrete configuration type declared by an environment factory."""
    for base in environment_type.__orig_bases__:
        if get_origin(base) is ExampleEnvironmentBase:
            (cfg_type,) = get_args(base)
            return cfg_type
    raise AssertionError(f"{environment_type.__name__} does not declare a typed environment configuration")


def test_all_registered_environments_translate_legacy_defaults_to_typed_cfg(monkeypatch):
    """Every registered factory must translate its legacy defaults to its concrete Cfg."""
    ensure_environments_registered()

    for environment_name in sorted(EnvironmentRegistry().get_all_keys()):
        environment_type = EnvironmentRegistry().get_component_by_name(environment_name)
        cfg_type = _environment_cfg_type(environment_type)
        assert is_dataclass(cfg_type), f"{cfg_type.__name__} must be a dataclass"

        parser = argparse.ArgumentParser(exit_on_error=False)
        environment_type.add_cli_args(parser)
        legacy_arguments = parser.parse_args([])
        legacy_arguments.enable_cameras = False
        legacy_arguments.mimic = False
        legacy_arguments.num_envs = 1

        captured = {}
        expected_environment = object()

        def fake_build(self, cfg):
            captured["cfg"] = cfg
            return expected_environment

        monkeypatch.setattr(environment_type, "build", fake_build)
        factory = object.__new__(environment_type)

        environment = factory.get_env(legacy_arguments)
        cfg = captured["cfg"]

        assert environment is expected_environment
        assert type(cfg) is cfg_type
        assert isinstance(cfg, ArenaEnvironmentCfg)
        expected_cfg = cfg_type(**{
            cfg_field.name: getattr(legacy_arguments, cfg_field.name)
            for cfg_field in fields(cfg)
            if hasattr(legacy_arguments, cfg_field.name)
        })
        assert cfg == expected_cfg, f"{environment_name} did not retain its legacy defaults"
