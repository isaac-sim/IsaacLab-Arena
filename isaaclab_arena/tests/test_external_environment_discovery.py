# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for externally packaged Arena environment discovery."""

from __future__ import annotations

import gymnasium as gym
from dataclasses import dataclass

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class _ExternalEnvironmentCfg(ArenaEnvironmentCfg):
    object: str = "cracker_box"


class _ExternalEnvironment(ArenaEnvironmentFactory[_ExternalEnvironmentCfg]):
    name = "external_environment_discovery_test"

    def build(self, cfg):
        raise NotImplementedError


class _ConflictingExternalEnvironment(ArenaEnvironmentFactory[_ExternalEnvironmentCfg]):
    name = _ExternalEnvironment.name

    def build(self, cfg):
        raise NotImplementedError


class _FakeEntryPoint:
    def __init__(self, name, loaded_value):
        self.name = name
        self._loaded_value = loaded_value

    def load(self):
        return self._loaded_value


@pytest.fixture
def clean_external_environment_registration():
    registry = EnvironmentRegistry()
    yield
    registry._components.pop(_ExternalEnvironment.name, None)
    registry._cfg_types_by_factory_type.pop(_ExternalEnvironment, None)
    registry._factory_types_by_cfg_type.pop(_ExternalEnvironmentCfg, None)
    gym.registry.pop(_ExternalEnvironment.name, None)


def _publish_external_environment(monkeypatch):
    from isaaclab_arena.environments import environment_discovery

    entry_point = _FakeEntryPoint(_ExternalEnvironment.name, _ExternalEnvironment)
    monkeypatch.setattr(environment_discovery.importlib.metadata, "entry_points", lambda **_kwargs: [entry_point])


def test_external_environment_entry_point_registers_factory_once(monkeypatch, clean_external_environment_registration):
    from isaaclab_arena.environments.environment_discovery import load_external_environment_entry_points

    _publish_external_environment(monkeypatch)

    load_external_environment_entry_points()
    load_external_environment_entry_points()

    registry = EnvironmentRegistry()
    assert registry.get_component_by_name(_ExternalEnvironment.name) is _ExternalEnvironment
    assert registry._cfg_types_by_factory_type[_ExternalEnvironment] is _ExternalEnvironmentCfg


def test_registered_entry_point_remains_compatible_with_class_path(
    monkeypatch, clean_external_environment_registration
):
    from isaaclab_arena.environments.environment_discovery import load_external_environment_entry_points
    from isaaclab_arena_environments import cli

    _publish_external_environment(monkeypatch)
    load_external_environment_entry_points()
    monkeypatch.setattr(
        cli,
        "parse_and_return_external_environment_from_string",
        lambda _path: (_ExternalEnvironment.name, _ExternalEnvironment),
    )

    name, factory_type = cli.register_external_environment_from_class_path("unused:test")

    assert name == _ExternalEnvironment.name
    assert factory_type is _ExternalEnvironment


def test_external_environment_entry_point_rejects_invalid_factory(monkeypatch, clean_external_environment_registration):
    from isaaclab_arena.environments import environment_discovery

    entry_point = _FakeEntryPoint("not_an_environment", object)
    monkeypatch.setattr(environment_discovery.importlib.metadata, "entry_points", lambda **_kwargs: [entry_point])

    with pytest.raises(TypeError, match="ArenaEnvironmentFactory"):
        environment_discovery.load_external_environment_entry_points()


def test_external_environment_entry_point_rejects_duplicate_names(monkeypatch, clean_external_environment_registration):
    from isaaclab_arena.environments import environment_discovery

    entry_points = [
        _FakeEntryPoint(_ExternalEnvironment.name, _ExternalEnvironment),
        _FakeEntryPoint(_ExternalEnvironment.name, _ExternalEnvironment),
    ]
    monkeypatch.setattr(environment_discovery.importlib.metadata, "entry_points", lambda **_kwargs: entry_points)

    with pytest.raises(RuntimeError, match="Multiple Arena environment entry points"):
        environment_discovery.load_external_environment_entry_points()


def test_external_environment_entry_point_rejects_registry_collision(
    monkeypatch, clean_external_environment_registration
):
    from isaaclab_arena.environments import environment_discovery

    EnvironmentRegistry().register(_ConflictingExternalEnvironment, _ExternalEnvironment.name)
    _publish_external_environment(monkeypatch)

    with pytest.raises(RuntimeError, match="already registered"):
        environment_discovery.load_external_environment_entry_points()


def test_external_environment_is_available_to_gym_discovery(monkeypatch, clean_external_environment_registration):
    from isaaclab_tasks.utils.task_registration import TASK_REGISTRATION_CALLBACK_KEY

    from isaaclab_arena.environments.isaaclab_interop import register_arena_tasks_for_discovery

    _publish_external_environment(monkeypatch)

    register_arena_tasks_for_discovery()

    spec = gym.spec(_ExternalEnvironment.name)
    assert spec.kwargs[TASK_REGISTRATION_CALLBACK_KEY].endswith(".environment_registration_callback")


def test_external_environment_is_available_to_typed_experiments(monkeypatch, clean_external_environment_registration):
    import isaaclab_arena.assets.registries as registries
    from isaaclab_arena.evaluation.arena_experiment_config_loader import _registered_environment_cfg_types

    _publish_external_environment(monkeypatch)
    monkeypatch.setattr(registries, "ensure_assets_registered", lambda: None)
    environment_cfg_types = _registered_environment_cfg_types()

    assert environment_cfg_types[_ExternalEnvironment.name] is _ExternalEnvironmentCfg
