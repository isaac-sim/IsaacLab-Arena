# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from abc import ABC
from dataclasses import MISSING, Field, fields
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args, get_origin, get_type_hints

from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

ArenaEnvironmentCfgT = TypeVar("ArenaEnvironmentCfgT", bound=ArenaEnvironmentCfg)

# Legacy argparse compatibility
#
# Environment config dataclasses are the source of truth. While the argparse frontend
# still exists, the helpers below generate environment-subcommand flags from those
# dataclasses. The root parser or a specialized runner already defines these shared
# fields, so environment subparsers must not add them a second time.
# TODO(cvolk, 2026-07-03): Delete this entire section when runner entry points receive
# typed environment configs directly.
_FIELDS_PROVIDED_BY_SHARED_PARSERS = {"auto", "enable_cameras", "mimic", "num_envs"}


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

    # Reuse the dataclass default so CLI and typed construction cannot drift apart.
    if config_field.default is not MISSING:
        argument_options["default"] = config_field.default
    elif config_field.default_factory is not MISSING:
        argument_options["default"] = config_field.default_factory()
    else:
        argument_options["required"] = True

    cli_value_type = _unwrap_optional_type(field_type)
    if get_origin(cli_value_type) is list:
        (list_item_type,) = get_args(cli_value_type)
        # All list flags use the same zero-or-more-values syntax.
        argument_options.update(type=list_item_type, nargs="*")
    elif cli_value_type is bool:
        if argument_options.get("default") is False:
            argument_options["action"] = "store_true"
        else:
            argument_options["action"] = argparse.BooleanOptionalAction
    else:
        argument_options["type"] = cli_value_type
    return argument_options


def _add_environment_config_arguments(
    parser: argparse.ArgumentParser,
    environment_cfg_type: type[ArenaEnvironmentCfg],
) -> None:
    """Generate one CLI flag for each environment-owned config field."""
    resolved_field_types = get_type_hints(environment_cfg_type)
    for config_field in fields(environment_cfg_type):
        field_name = config_field.name
        if field_name in _FIELDS_PROVIDED_BY_SHARED_PARSERS:
            continue
        argument_options = _get_argparse_options(config_field, resolved_field_types[field_name])
        argument_options["help"] = f"{field_name.replace('_', ' ')} (default: %(default)s)"
        parser.add_argument(f"--{field_name}", **argument_options)


# TODO(cvolk, 2026-07-03): Co-locate ArenaEnvironmentCfg and this base in the core
# arena_environment_factory module as ArenaEnvironmentFactoryBase.
class ExampleEnvironmentBase(ABC, Generic[ArenaEnvironmentCfgT]):
    """Initialize shared registries and bridge legacy argparse into ``build(cfg)``."""

    name: str | None = None

    # Generic arguments are for static type checking. The legacy adapter also needs the
    # concrete dataclass at runtime so it can inspect its fields.
    # TODO(cvolk, 2026-07-03): Delete this metadata with get_env() and add_cli_args().
    _legacy_argparse_cfg_type: type[ArenaEnvironmentCfgT] | None = None

    def __init__(self):
        from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry, HDRImageRegistry

        self.asset_registry = AssetRegistry()
        self.device_registry = DeviceRegistry()
        self.hdr_registry = HDRImageRegistry()

    # TODO(cvolk, 2026-07-03): Delete this Namespace adapter with the legacy argparse route.
    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        """Translate the legacy CLI namespace and build the environment."""
        environment_cfg_type = self._legacy_argparse_cfg_type
        assert environment_cfg_type is not None, f"{type(self).__name__} must define _legacy_argparse_cfg_type"

        # Some runner-specific fields are not present in every Namespace. Copy the values
        # that are present and let the dataclass supply defaults for the rest.
        config_values = {
            config_field.name: getattr(args_cli, config_field.name)
            for config_field in fields(environment_cfg_type)
            if hasattr(args_cli, config_field.name)
        }
        return self.build(environment_cfg_type(**config_values))

    def build(self, cfg: ArenaEnvironmentCfgT) -> IsaacLabArenaEnvironment:
        """Build an Arena environment from its typed configuration."""
        raise NotImplementedError(f"{type(self).__name__} does not support typed environment configuration")

    # TODO(cvolk, 2026-07-03): Delete this generated-flag entry point with the legacy argparse route.
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Generate legacy CLI arguments from the typed environment config."""
        environment_cfg_type = cls._legacy_argparse_cfg_type
        assert environment_cfg_type is not None, f"{cls.__name__} must define _legacy_argparse_cfg_type"
        _add_environment_config_arguments(parser, environment_cfg_type)
