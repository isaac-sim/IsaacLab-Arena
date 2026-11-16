# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the workbench cleanup scene and tasks."""

from pathlib import Path

import pytest

from isaaclab_arena.agentic_tools.object_control.object_control_recipe import (
    CheckCommand,
    MarkCommand,
    load_object_control_recipe,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_ENVIRONMENT_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/workbench_cleanup"
_TASK_PATHS = sorted(_ENVIRONMENT_DIR.glob("task_*.yaml"))


def _test_workbench_task_definitions_build(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

    for path in sorted(_ENVIRONMENT_DIR.glob("task_*.yaml")):
        environment = build_arena_env_from_graph_spec(ArenaEnvGraphSpec.from_yaml(path))
        assert environment.name
    return True


def test_workbench_task_definitions_build():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_task_definitions_build)


@pytest.mark.parametrize("task_path", _TASK_PATHS, ids=[path.stem for path in _TASK_PATHS])
def test_workbench_task_and_recipe_load(task_path):
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    recipe_name = task_path.stem.removeprefix("task_")
    recipe = load_object_control_recipe(_ENVIRONMENT_DIR / "object_control" / f"{recipe_name}.jsonl")

    subtask_indices = {command.index for command in recipe.commands if isinstance(command, (MarkCommand, CheckCommand))}
    assert subtask_indices == set(range(len(spec.task.subtasks)))
