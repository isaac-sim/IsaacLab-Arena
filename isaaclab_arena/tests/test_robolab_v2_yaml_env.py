# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: RoboLab v2 task YAMLs build and reset in sim."""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

_ROBOLAB_V2_TASKS_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "robolab_v2" / "tasks"
_ROBOLAB_V2_TASK_YAMLS = sorted(_ROBOLAB_V2_TASKS_DIR.glob("*.yaml"))


def _test_robolab_v2_yaml_env_bringup(simulation_app, *, yaml_path: Path) -> bool:
    """Build an env from a RoboLab v2 task YAML and reset once."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    env = None
    try:
        spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
        arena_env = spec.to_arena_env()
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--placement_seed", "0"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()

        assert arena_env.name == spec.env_name
        assert "robot" in env.unwrapped.scene.keys(), f"robot missing from scene for {yaml_path.name}"
    except Exception as exc:
        print(f"Error bringing up {yaml_path}: {exc}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            env.close()

    return True


def test_robolab_v2_has_tasks():
    assert _ROBOLAB_V2_TASK_YAMLS, f"no RoboLab v2 task YAMLs found in {_ROBOLAB_V2_TASKS_DIR}"


@pytest.mark.parametrize(
    "yaml_path",
    _ROBOLAB_V2_TASK_YAMLS,
    ids=[path.stem for path in _ROBOLAB_V2_TASK_YAMLS],
)
def test_robolab_v2_yaml_env_bringup(yaml_path: Path):
    """Each RoboLab v2 task YAML must parse and produce a resettable Arena env."""
    result = run_function_with_persistent_simulation_app(
        _test_robolab_v2_yaml_env_bringup,
        headless=True,
        yaml_path=yaml_path,
    )
    assert result, f"RoboLab v2 bring-up failed for {yaml_path.name}"


if __name__ == "__main__":
    for path in _ROBOLAB_V2_TASK_YAMLS:
        test_robolab_v2_yaml_env_bringup(path)
