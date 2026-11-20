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
import textwrap
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import add_external_environments_cli_args
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena_environments.cli import (
    _environment_cfg_from_cli,
    add_environment_cli_args,
    add_example_environments_cli_args,
    build_environment_from_cli,
    ensure_environments_registered,
    import_external_environment_factory,
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

        environment_cfg_type = environment_registry.get_environment_cfg_types()[environment_name]
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


# Decorated factories from these modules stay in EnvironmentRegistry after their test. Their config
# gives every field a default, so tests that parse each registered environment's default flags pass.
_EXTERNAL_FACTORY_MODULE_TEMPLATE = """
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class ExternalEnvironmentCfg(ArenaEnvironmentCfg):
    object: str = "cracker_box"


{decorator}
class ExternalEnvironment(ArenaEnvironmentFactory[ExternalEnvironmentCfg]):
    name = "{environment_name}"

    def build(self, cfg):
        raise NotImplementedError
"""


def _write_external_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, source: str) -> None:
    """Write an importable module that only the code under test imports."""
    (tmp_path / f"{module_name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)
    assert module_name not in sys.modules


def _write_external_factory_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    environment_name: str,
    decorate: bool = True,
) -> str:
    """Write a module with one external environment factory and return its ``module:Class`` path."""
    source = _EXTERNAL_FACTORY_MODULE_TEMPLATE.format(
        decorator="@register_environment" if decorate else "",
        environment_name=environment_name,
    )
    _write_external_module(tmp_path, monkeypatch, module_name, source)
    return f"{module_name}:ExternalEnvironment"


def test_external_environment_flag_imports_the_registered_factory(tmp_path, monkeypatch):
    """Register an external factory only through --external_environment_class_path and parse its flags."""
    environment_name = "flag_only_external_environment"
    environment_class_path = _write_external_factory_module(
        tmp_path, monkeypatch, "flag_only_external_environment_module", environment_name
    )
    ensure_environments_registered()
    assert not EnvironmentRegistry().is_registered(environment_name)

    cli_args = ["--external_environment_class_path", environment_class_path, environment_name, "--object", "mug"]
    monkeypatch.setattr(sys, "argv", ["environment_runner.py", *cli_args])
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_external_environments_cli_args(parser)
    add_example_environments_cli_args(parser)
    legacy_arguments = parser.parse_args(cli_args)

    external_module = sys.modules["flag_only_external_environment_module"]
    environment_factory_type = EnvironmentRegistry().get_component_by_name(environment_name)
    assert environment_factory_type is external_module.ExternalEnvironment
    assert legacy_arguments.example_environment == environment_name
    environment_cfg = _environment_cfg_from_cli(environment_factory_type, legacy_arguments)
    assert environment_cfg == external_module.ExternalEnvironmentCfg(object="mug")


def test_external_environment_flag_returns_the_same_factory_on_repeated_imports(tmp_path, monkeypatch):
    """Return the registered factory again when the same class path is imported a second time."""
    environment_class_path = _write_external_factory_module(
        tmp_path, monkeypatch, "repeated_external_environment_module", "repeated_external_environment"
    )

    first_factory_type = import_external_environment_factory(environment_class_path)
    second_factory_type = import_external_environment_factory(environment_class_path)

    assert second_factory_type is first_factory_type
    assert EnvironmentRegistry().get_component_by_name("repeated_external_environment") is first_factory_type


def test_external_environment_flag_rejects_an_unregistered_factory(tmp_path, monkeypatch):
    """Ask for @register_environment instead of registering the factory without its config."""
    environment_class_path = _write_external_factory_module(
        tmp_path,
        monkeypatch,
        "unregistered_external_environment_module",
        "unregistered_external_environment",
        decorate=False,
    )

    with pytest.raises(AssertionError, match="is not registered. Decorate it with @register_environment"):
        import_external_environment_factory(environment_class_path)
    assert not EnvironmentRegistry().is_registered("unregistered_external_environment")


def test_external_environment_flag_rejects_a_name_taken_by_another_factory(tmp_path, monkeypatch):
    """Name the factory that already owns a name instead of replacing it."""
    environment_class_path = _write_external_factory_module(
        tmp_path, monkeypatch, "taken_name_external_environment_module", "kitchen_pick_and_place"
    )

    with pytest.raises(
        AssertionError,
        match=(
            "uses the name 'kitchen_pick_and_place', which is taken by"
            r" isaaclab_arena_environments\.kitchen_pick_and_place_environment\.KitchenPickAndPlaceEnvironment"
        ),
    ):
        import_external_environment_factory(environment_class_path)


def test_external_environment_flag_rejects_a_class_that_is_not_a_factory(tmp_path, monkeypatch):
    """Reject a class that does not subclass ArenaEnvironmentFactory."""
    _write_external_module(
        tmp_path,
        monkeypatch,
        "non_factory_external_environment_module",
        """
        class NotAnEnvironmentFactory:
            name = "non_factory_external_environment"
        """,
    )

    with pytest.raises(AssertionError, match="is not an ArenaEnvironmentFactory subclass"):
        import_external_environment_factory("non_factory_external_environment_module:NotAnEnvironmentFactory")


def test_maple_teleop_device_remains_a_cli_only_option():
    """Keep Maple's teleop selection out of the environment construction config."""
    legacy_arguments = _parse_legacy_arguments(
        PickAndPlaceMapleTableEnvironment,
        ["--teleop_device", "spacemouse"],
    )
    environment_cfg = _environment_cfg_from_cli(PickAndPlaceMapleTableEnvironment, legacy_arguments)

    assert legacy_arguments.teleop_device == "spacemouse"
    assert "teleop_device" not in {config_field.name for config_field in fields(environment_cfg)}
