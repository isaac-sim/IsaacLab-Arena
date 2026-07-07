# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Format values used only by the existing JSON evaluation-job frontend."""

from dataclasses import dataclass
from typing import Any

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

# TODO(cvolk, 2026-07-07): Delete this module with the JSON evaluation-job
# frontend once eval_runner loads typed YAML experiments directly.


@dataclass
class LegacyGraphEnvironmentCfg(ArenaEnvironmentCfg):
    """Carry a graph-YAML environment through its temporary CLI construction path."""

    arena_env_args: list[str]
    """Arguments consumed by the existing graph-environment parser."""


def legacy_environment_args_to_cli_args(args: dict[str, Any]) -> list[str]:
    """Convert legacy environment arguments into the existing parser's token order."""
    assert args.get("environment") is not None, "environment is required in legacy environment arguments"

    cli_args = []
    priority_keys = ("num_envs", "env_spacing", "enable_cameras", "placement_seed")
    for key in priority_keys:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, bool) and value:
            cli_args.append(f"--{key}")
        elif not isinstance(value, bool) and value is not None:
            cli_args.extend((f"--{key}", str(value)))

    environment = str(args["environment"])
    if environment.endswith((".yaml", ".yml")):
        cli_args.extend(("--env_graph_spec_yaml", environment))
    else:
        cli_args.append(environment)

    for key, value in args.items():
        if key in priority_keys or key == "environment":
            continue
        if isinstance(value, bool) and value:
            cli_args.append(f"--{key}")
        elif not isinstance(value, bool) and value is not None:
            cli_args.extend((f"--{key}", str(value)))
    return cli_args
