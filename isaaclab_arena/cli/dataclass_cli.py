# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate temporary argparse interfaces from typed dataclass configs."""

from __future__ import annotations

import argparse
from collections.abc import Collection
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

CfgT = TypeVar("CfgT")

# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this module when Arena frontends receive typed
# configuration objects instead of reconstructing them from argparse Namespaces.


def add_dataclass_cli_args(
    parser: argparse.ArgumentParser,
    cfg_type: type[CfgT],
    *,
    excluded_fields: Collection[str] = (),
) -> None:
    """Generate CLI flags for the selected fields of a config dataclass."""
    assert is_dataclass(cfg_type), f"{cfg_type.__name__} must be a dataclass"
    resolved_field_types = get_type_hints(cfg_type)
    for config_field in fields(cfg_type):
        if config_field.name in excluded_fields:
            continue

        argument_options = _get_argparse_options(config_field, resolved_field_types[config_field.name])
        if argument_options.get("required", False):
            argument_options["help"] = config_field.name.replace("_", " ")
        else:
            argument_options["help"] = f"{config_field.name.replace('_', ' ')} (default: %(default)s)"
        parser.add_argument(f"--{config_field.name}", **argument_options)


def assert_cli_defaults_match_dataclass(
    parser: argparse.ArgumentParser,
    cfg_type: type[CfgT],
    field_names: Collection[str],
) -> None:
    """Assert that existing parser flags retain their dataclass defaults."""
    assert is_dataclass(cfg_type), f"{cfg_type.__name__} must be a dataclass"
    config_fields = {config_field.name: config_field for config_field in fields(cfg_type)}
    for field_name in field_names:
        assert field_name in config_fields, f"{cfg_type.__name__} has no field {field_name}"
        option = f"--{field_name}"
        assert option in parser._option_string_actions, f"The shared parser must define {option}"

        cfg_default = _field_default(config_fields[field_name])
        assert cfg_default is not MISSING, f"Shared config field {field_name} must have a default"
        parser_default = parser._option_string_actions[option].default
        assert (
            parser_default == cfg_default
        ), f"{cfg_type.__name__}.{field_name} defaults to {cfg_default!r}, but {option} defaults to {parser_default!r}"


def dataclass_from_cli(cfg_type: type[CfgT], args_cli: argparse.Namespace) -> CfgT:
    """Create a config dataclass from matching values in an argparse Namespace."""
    assert is_dataclass(cfg_type), f"{cfg_type.__name__} must be a dataclass"
    config_values = {
        config_field.name: getattr(args_cli, config_field.name)
        for config_field in fields(cfg_type)
        if hasattr(args_cli, config_field.name)
    }
    return cfg_type(**config_values)


def _field_default(config_field: Field[Any]) -> Any:
    """Return one dataclass field's default value, or ``MISSING``."""
    if config_field.default is not MISSING:
        return config_field.default
    if config_field.default_factory is not MISSING:
        return config_field.default_factory()
    return MISSING


def _unwrap_optional_type(field_type: Any) -> Any:
    """Return ``T`` from ``T | None`` so argparse can use ``T`` as a converter."""
    union_members = get_args(field_type)
    non_none_members = tuple(member for member in union_members if member is not type(None))
    if len(non_none_members) == 1 and len(non_none_members) != len(union_members):
        return non_none_members[0]
    return field_type


def _get_argparse_options(config_field: Field[Any], field_type: Any) -> dict[str, Any]:
    """Describe one generated argparse flag using its dataclass field."""
    argument_options: dict[str, Any] = {}
    default = _field_default(config_field)
    if default is MISSING:
        argument_options["required"] = True
    else:
        argument_options["default"] = default

    cli_value_type = _unwrap_optional_type(field_type)
    if get_origin(cli_value_type) is list:
        (list_item_type,) = get_args(cli_value_type)
        argument_options.update(type=list_item_type, nargs="*")
    elif get_origin(cli_value_type) is Literal:
        choices = get_args(cli_value_type)
        choice_types = {type(choice) for choice in choices}
        assert len(choice_types) == 1, f"{config_field.name} Literal choices must have one value type"
        argument_options.update(type=choice_types.pop(), choices=choices)
    elif cli_value_type is bool:
        argument_options["action"] = "store_true" if default is False else argparse.BooleanOptionalAction
    else:
        argument_options["type"] = cli_value_type
    return argument_options
