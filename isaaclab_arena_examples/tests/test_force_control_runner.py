# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest


def test_load_force_control_recipe_separates_setup_from_runtime_commands(tmp_path):
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import load_force_control_recipe

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


def test_joint_setup_can_keep_a_passive_return_spring():
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import (
        _apply_joint_setups,
        _validate_command,
    )

    command = {
        "action": "setup_joint",
        "name": "scope",
        "joint_name": "button",
        "drive_stiffness": 800.0,
        "drive_damping": 0.5,
        "effort_limit": 3.0,
    }
    cfg = SimpleNamespace(scene=SimpleNamespace(scope=SimpleNamespace(actuators={})))
    _apply_joint_setups(cfg, (_validate_command(command, 1),))
    actuator = cfg.scene.scope.actuators["force_control_button"]
    assert actuator.stiffness == 800.0
    assert actuator.damping == 0.5
    assert actuator.effort_limit_sim == 3.0


@pytest.mark.parametrize("field", ["drive_stiffness", "drive_damping"])
def test_joint_setup_rejects_negative_passive_drive_gains(field):
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import _validate_command

    with pytest.raises(AssertionError, match="non-negative"):
        _validate_command({"action": "setup_joint", "name": "scope", "joint_name": "button", field: -1}, 1)


@pytest.mark.parametrize("hold", [None, {"gain": 2.0, "damping": 0.015, "limit": 0.01}])
def test_optional_reset_joint_hold_is_active_before_initial_settling(hold):
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import (
        ForceControlRecipe,
        PhysicalCommandRunner,
    )

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    reset_joint_positions = torch.tensor([[0.7, -0.001]])
    runner.env = SimpleNamespace(
        scene={
            "scope": SimpleNamespace(
                data=SimpleNamespace(
                    joint_names=["handle", "button"], joint_pos=SimpleNamespace(torch=reset_joint_positions)
                )
            )
        }
    )
    runner.args = Namespace(follow_commands=False, allow_incomplete=False)
    runner.writer = None
    runner.joint_holds = {}
    observed = []
    runner.step = lambda count: observed.append((count, runner.joint_holds.copy()))
    runner._execute_finite = lambda commands: None
    runner._write_report = lambda: {"success": [True]}
    setup = {"action": "setup_joint", "name": "scope", "joint_name": "button"}
    if hold is not None:
        setup["settle_hold"] = hold

    runner.run(ForceControlRecipe((setup,), ()))

    assert len(observed) == 1 and observed[0][0] == 240
    if hold is None:
        assert observed[0][1] == {}
    else:
        assert observed[0][1] == {("scope", "button"): pytest.approx((-0.001, 2.0, 0.015, 0.01))}
    assert torch.equal(reset_joint_positions, torch.tensor([[0.7, -0.001]]))


@pytest.mark.parametrize(
    "hold",
    [True, {}, {"gain": 2, "damping": 0, "limit": -1}, {"gain": float("nan"), "damping": 0, "limit": 1}],
)
def test_validate_command_rejects_invalid_settle_hold(hold):
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import _validate_command

    with pytest.raises(AssertionError, match="settle_hold"):
        _validate_command({"action": "setup_joint", "name": "scope", "joint_name": "button", "settle_hold": hold}, 1)


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
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import _validate_command

    with pytest.raises(AssertionError, match=message):
        _validate_command(command, 4)


def test_combine_reference_goal_uses_only_selected_reference_axes():
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import _combine_reference_goal

    result = _combine_reference_goal(
        goal=[1.0, 2.0, 0.56],
        reference=[0.2, -0.1, 0.8],
        offset=[0.05, 0.06, 0.0],
        reference_axes=["x", "y"],
    )

    assert result == pytest.approx([0.25, -0.04, 0.56])


def test_criterion_status_evaluates_wrapped_and_direct_predicates():
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

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


def test_contact_sensor_state_discovers_configured_contact_sensors():
    import torch

    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

    contact_cfg = ContactSensorCfg(
        prim_path="/World/Object",
        filter_prim_paths_expr=["/World/Support"],
    )
    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(
        cfg=SimpleNamespace(scene=SimpleNamespace(object_contact=contact_cfg, unrelated=object())),
        scene=SimpleNamespace(
            sensors={
                "object_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        net_forces_w=torch.tensor([[[1.0, 2.0, 3.0]]]),
                        force_matrix_w=torch.tensor([[[[4.0, 5.0, 6.0]]]]),
                    )
                )
            }
        ),
    )

    assert runner._contact_sensor_state() == {
        "object_contact": {
            "prim_path": "/World/Object",
            "filter_prim_paths_expr": ["/World/Support"],
            "net_forces_w": [[[
                1.0,
                2.0,
                3.0,
            ]]],
            "force_matrix_w": [[[[
                4.0,
                5.0,
                6.0,
            ]]]],
        }
    }


def test_check_timeout_writes_failure_snapshot(tmp_path):
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.output = tmp_path
    runner.steps = 10
    runner.viewport = None
    runner.last_image = None
    runner._task_completion = lambda: torch.tensor([[False]])
    runner._criterion_status = lambda: {"criterion": [[False]]}
    runner._collect_state = lambda: {"completion": [[False]], "current_predicates": {"criterion": [[False]]}}
    runner.step = lambda: setattr(runner, "steps", runner.steps + 1)

    with pytest.raises(AssertionError, match="diagnostics:"):
        runner.check(0, timeout_steps=2)

    [snapshot_path] = tmp_path.glob("failure_check_timeout_subtask_00_step_*.json")
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["trigger"] == "check_timeout"
    assert snapshot["completion"] == [[False]]


def test_execute_dispatches_commands_and_stops_on_finish(monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

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

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

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
    from isaaclab_arena_examples.agentic_environment_generation import force_control_runner

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


def test_force_steps_apply_interval_events_at_control_rate():
    """Manual physics stepping must also advance configured scene events."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from isaaclab_arena_examples.agentic_environment_generation.force_control_runner import PhysicalCommandRunner

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    events = SimpleNamespace(available_modes=["interval"], apply=Mock())
    runner.env = SimpleNamespace(
        scene=SimpleNamespace(write_data_to_sim=Mock(), update=Mock()),
        sim=SimpleNamespace(step=Mock()),
        cfg=SimpleNamespace(decimation=2),
        physics_dt=0.01,
        step_dt=0.02,
        common_step_counter=0,
        episode_length_buf=0,
        update_task_progress=Mock(),
        event_manager=events,
    )
    runner._apply_forces = Mock()
    runner.steps = 0
    runner.viewport = None
    runner.step(5)
    assert events.apply.call_count == 2
    events.apply.assert_called_with(mode="interval", dt=0.02)
    assert runner.env.common_step_counter == 2
