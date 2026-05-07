# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register an Arena environment, then delegate RL-Games training to Isaac Lab.

This file is intentionally a thin compatibility wrapper. The training loop,
checkpoint handling, video recording, and RL-Games runner setup live in Isaac
Lab's ``scripts/reinforcement_learning/rl_games/train.py``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback

_FORWARDED_VALUE_OPTIONS = {
    "--agent",
    "--checkpoint",
    "--device",
    "--max_iterations",
    "--num_envs",
    "--ray-proc-id",
    "--seed",
    "--sigma",
    "--task",
    "--video_interval",
    "--video_length",
    "--wandb-entity",
    "--wandb-name",
    "--wandb-project-name",
}
_FORWARDED_FLAG_OPTIONS = {
    "--distributed",
    "--export_io_descriptors",
    "--headless",
    "--track",
    "--video",
}
_DEPRECATED_VALUE_OPTIONS = {
    "--agent_cfg_path",
    "--experiment_name",
}


def _normalize_legacy_positional_task(argv: list[str]) -> list[str]:
    if "--task" in argv or any(arg.startswith("--task=") for arg in argv):
        return argv

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()

    for idx, arg in enumerate(argv):
        if not arg.startswith("-") and env_registry.is_registered(arg):
            print("[WARN] Positional environment names are deprecated for RL-Games training; use --task instead.")
            return [*argv[:idx], "--task", arg, *argv[idx + 1 :]]
    return argv


def _isaac_lab_rl_games_train_script() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "submodules" / "IsaacLab" / "scripts" / "reinforcement_learning" / "rl_games" / "train.py"


def _copy_forwarded_args(argv: list[str]) -> list[str]:
    forwarded: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        option = arg.split("=", maxsplit=1)[0]
        if option in _FORWARDED_FLAG_OPTIONS:
            forwarded.append(arg)
        elif option in _FORWARDED_VALUE_OPTIONS:
            forwarded.append(arg)
            if "=" not in arg and idx + 1 < len(argv):
                idx += 1
                forwarded.append(argv[idx])
        elif option in _DEPRECATED_VALUE_OPTIONS and "=" not in arg and idx + 1 < len(argv):
            idx += 1
        idx += 1
    return forwarded


def _drop_forwarded_args(argv: list[str]) -> list[str]:
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        option = arg.split("=", maxsplit=1)[0]
        if option in _FORWARDED_FLAG_OPTIONS:
            idx += 1
            continue
        if option in _FORWARDED_VALUE_OPTIONS:
            if "=" not in arg and idx + 1 < len(argv):
                idx += 1
            idx += 1
            continue
        remaining.append(arg)
        idx += 1
    return remaining


def _remove_deprecated_args(argv: list[str]) -> tuple[list[str], str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent_cfg_path", default=None)
    parser.add_argument("--experiment_name", default=None)
    deprecated, remaining = parser.parse_known_args(argv)
    if deprecated.agent_cfg_path is not None:
        print("[WARN] --agent_cfg_path is ignored; the registered Arena environment provides rl_games_cfg_entry_point.")
    return remaining, deprecated.experiment_name


def main() -> None:
    original_args = _normalize_legacy_positional_task(sys.argv[1:])
    sys.argv = [sys.argv[0], *original_args]
    lab_script = _isaac_lab_rl_games_train_script()
    if not lab_script.is_file():
        raise FileNotFoundError(f"Isaac Lab RL-Games trainer not found: {lab_script}")

    remaining_args = environment_registration_callback()
    remaining_args, experiment_name = _remove_deprecated_args(remaining_args)
    remaining_args = _drop_forwarded_args(remaining_args)

    forwarded_args = _copy_forwarded_args(original_args)
    if experiment_name is not None:
        forwarded_args.append(f"agent.params.config.name={experiment_name}")

    sys.argv = [str(lab_script), *forwarded_args, *remaining_args]
    runpy.run_path(str(lab_script), run_name="__main__")


if __name__ == "__main__":
    main()
