# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest


def test_load_force_control_recipe_separates_setup_from_runtime_commands(tmp_path):
    from isaaclab_arena.scripts.force_control_runner import load_force_control_recipe

    recipe_path = tmp_path / "recipe.jsonl"
    commands = [
        {
            "action": "setup_joint",
            "name": "drawer",
            "joint_name": "slide",
            "effort_limit": 50.0,
        },
        {"action": "joint", "name": "drawer", "joint_name": "slide", "goal": 0.2},
        {"action": "finish"},
    ]
    recipe_path.write_text("\n".join(json.dumps(command) for command in commands))

    recipe = load_force_control_recipe(recipe_path)

    assert recipe.joint_setups == (commands[0],)
    assert recipe.commands == tuple(commands[1:])


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({"action": "unknown"}, "unknown action"),
        ({"action": "move", "name": "cube"}, "requires 'goal' or 'reference'"),
        (
            {"action": "move", "name": "cube", "goal": [1, 2, 3], "extra": True},
            "unknown fields",
        ),
        ({"action": "wait", "steps": 1, "seconds": 0.1}, "exactly one"),
        ({"action": "joint", "name": "door", "joint_name": "hinge"}, "exactly one"),
        ({"action": "joint", "name": "door", "joint_name": "hinge", "goal": [0.1]}, "must be a number"),
        (
            {"action": "joint", "name": "door", "joint_name": "hinge", "normalized_goal": 1.1},
            "between 0 and 1",
        ),
    ],
)
def test_validate_command_rejects_invalid_recipes(command, message):
    from isaaclab_arena.scripts.force_control_runner import _validate_command

    with pytest.raises(AssertionError, match=message):
        _validate_command(command, 4)


def test_combine_reference_goal_uses_only_selected_reference_axes():
    from isaaclab_arena.scripts.force_control_runner import _combine_reference_goal

    result = _combine_reference_goal(
        goal=[1.0, 2.0, 0.56],
        reference=[0.2, -0.1, 0.8],
        offset=[0.05, 0.06, 0.0],
        reference_axes=["x", "y"],
    )

    assert result == pytest.approx([0.25, -0.04, 0.56])


def test_criterion_status_evaluates_wrapped_and_direct_predicates():
    import torch

    from isaaclab_arena.scripts.force_control_runner import PhysicalCommandRunner

    direct_predicate = lambda env: torch.tensor([True])  # noqa: E731
    wrapped_predicate = SimpleNamespace(predicate=lambda env: torch.tensor([False]))
    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(
        cfg=SimpleNamespace(
            task_progress=SimpleNamespace(
                success_criteria=[
                    SimpleNamespace(
                        name="criterion",
                        predicate_sequence=[direct_predicate, wrapped_predicate],
                    )
                ]
            )
        )
    )

    assert runner._criterion_status() == {"criterion": [[True], [False]]}


def test_execute_dispatches_commands_and_stops_on_finish(monkeypatch):
    from isaaclab_arena.scripts.force_control_runner import PhysicalCommandRunner

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    received = []
    monkeypatch.setattr(runner, "move", lambda command: received.append(command))

    command = {"action": "move", "name": "cube", "goal": [0.0, 0.0, 1.0]}
    assert runner._execute(command)
    assert received == [command]
    assert not runner._execute({"action": "finish"})


def test_overlay_progress_banner_draws_title_and_preserves_scene_pixels():
    import numpy as np
    import torch

    from isaaclab_arena.scripts.force_control_runner import PhysicalCommandRunner

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.current_subtask_index = 1
    runner.current_subtask_title = "Put the second banana in the bowl"
    runner._task_completion = lambda: torch.tensor([[True, False]])
    frame = np.full((120, 320, 3), 200, dtype=np.uint8)

    annotated = runner._overlay_progress_banner(frame)

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated[:76], frame[:76])
    assert np.array_equal(annotated[76:], frame[76:])


def test_run_recipe_closes_environment_when_execution_fails(monkeypatch):
    from isaaclab_arena.scripts import force_control_runner

    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(enable_scene_query_support=False),
        recorders=object(),
        episode_recorders=object(),
    )

    class FakeEnvironment:
        def __init__(self):
            self.unwrapped = object()
            self.reset_count = 0
            self.close_count = 0

        def reset(self):
            self.reset_count += 1

        def close(self):
            self.close_count += 1

    env = FakeEnvironment()
    builder = SimpleNamespace(
        build_env_cfg=lambda: (env_cfg, {"example": True}),
        make_registered=lambda cfg, kwargs: env,
    )

    class FailingRunner:
        def __init__(self, unwrapped, args):
            assert unwrapped is env.unwrapped

        def run(self, recipe):
            raise RuntimeError("recipe failed")

    monkeypatch.setattr(force_control_runner, "PhysicalCommandRunner", FailingRunner)
    recipe = force_control_runner.ForceControlRecipe((), ({"action": "finish"},))

    with pytest.raises(RuntimeError, match="recipe failed"):
        force_control_runner._run_recipe(builder, Namespace(), recipe)

    assert env_cfg.sim.enable_scene_query_support
    assert env_cfg.recorders == {}
    assert env_cfg.episode_recorders == {}
    assert env.reset_count == 1
    assert env.close_count == 1
