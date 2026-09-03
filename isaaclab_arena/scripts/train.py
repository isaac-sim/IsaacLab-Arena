# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train an RL policy with Isaac Lab's unified entry point and an Arena environment."""

from __future__ import annotations

import sys

# Warp captures ``enable_backward`` when modules are imported. Isaac Lab does not use
# Warp autodiff during RL training, so match its unified training entry point here.
import warp as wp

wp.config.enable_backward = False


_ARENA_REGISTRATION_CALLBACK = "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback"


def _get_cli_option(argv: list[str], option: str) -> str | None:
    """Return the last value supplied for a command-line option."""
    value = None
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(f"{option}="):
            value = token[len(option) + 1 :]
    return value


def _register_arena_task_for_rl_discovery(argv: list[str]) -> None:
    """Expose an Arena task to Isaac Lab's pre-callback agent discovery."""
    callback = _get_cli_option(argv, "--external_callback")
    if callback != _ARENA_REGISTRATION_CALLBACK:
        return

    environment_name = _get_cli_option(argv, "--task")
    if environment_name is None:
        return

    # Register native Isaac Lab tasks first so this bridge never shadows one.
    import isaaclab_tasks  # noqa: F401

    from isaaclab_arena.environments.isaaclab_interop import register_rl_discovery_placeholder

    register_rl_discovery_placeholder(environment_name)


def main(argv: list[str] | None = None) -> int:
    """Run Isaac Lab's unified trainer with Arena task discovery enabled."""
    if argv is None:
        argv = sys.argv[1:]
    _register_arena_task_for_rl_discovery(argv)

    from isaaclab_rl.entrypoints import run_train_cli

    return run_train_cli(argv)


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
