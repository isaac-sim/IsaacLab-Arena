# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import torch
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_load_robot_control_recipe_accepts_required_commands(tmp_path):
    from isaaclab_arena.scripts.robot_control_runner import load_robot_control_recipe

    commands = [
        {"action": "pose", "position": [0.4, 0.1, 0.5], "orientation": [0, 0, 0, 1], "gripper": 0},
        {
            "action": "pose",
            "reference": "tool",
            "reference_point": "aabb_center",
            "offset": [0, 0, 0.2],
            "orientation": [0, 0, 0, 1],
        },
        {"action": "gripper", "value": 1},
        {"action": "trajectory", "values": [[0.4, 0.1, 0.5, 0, 0, 0, 1, 1]]},
        {"action": "wait", "steps": 2},
        {"action": "mark", "index": 0, "boundary": "start", "title": "Move tool"},
        {"action": "check", "index": 0},
        {"action": "wait_final"},
        {"action": "inspect"},
        {"action": "finish"},
    ]
    path = tmp_path / "recipe.jsonl"
    path.write_text("\n".join(json.dumps(command) for command in commands))

    assert load_robot_control_recipe(path) == tuple(commands)


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({"action": "unknown"}, "unknown action"),
        ({"action": "pose"}, "requires 'values', 'position', or 'reference'"),
        ({"action": "pose", "values": [0] * 8}, "unit XYZW quaternion"),
        ({"action": "pose", "values": [0, 0, 0, 0, 0, 0, 1, 0.5]}, "gripper value must be 0 or 1"),
        ({"action": "gripper", "value": "open"}, "'value' must be 0 or 1"),
        ({"action": "trajectory", "waypoints": [], "values": []}, "exactly one"),
        ({"action": "trajectory", "waypoints": []}, "non-empty list"),
        ({"action": "wait", "steps": 1, "seconds": 0.1}, "exactly one"),
        (
            {"action": "pose", "reference": "cube", "orientation": [0, 0, 0, 1], "offset_frame": "robot"},
            "'offset_frame' must be 'reference' or 'world'",
        ),
    ],
)
def test_validate_command_rejects_invalid_recipes(command, message):
    from isaaclab_arena.scripts.robot_control_runner import _validate_command

    with pytest.raises(AssertionError, match=message):
        _validate_command(command, 4)


def test_world_pose_to_root_converts_translation_and_orientation():
    from isaaclab_arena.scripts.robot_control_runner import _world_pose_to_root

    half_sqrt_two = 2**-0.5
    T_W_R = np.array([1, 2, 0, 0, 0, half_sqrt_two, half_sqrt_two], dtype=np.float64)
    t_R_G, q_R_G = _world_pose_to_root(
        T_W_R,
        np.array([1, 3, 0], dtype=np.float64),
        np.array([0, 0, half_sqrt_two, half_sqrt_two], dtype=np.float64),
    )

    np.testing.assert_allclose(t_R_G, [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(q_R_G, [0, 0, 0, 1], atol=1e-7)


def test_scene_reference_target_rotates_offset_and_converts_to_robot_root():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner
    from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices

    half_sqrt_two = 2**-0.5
    poses = {
        "tool": torch.tensor([[1.0, 3.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two]]),
        "robot": torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two]]),
    }
    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.indices = AbsoluteCommandIndices(
        8,
        (PoseCommandIndices(slice(0, 7), "robot", "base_link"),),
        (),
        (GripperCommandIndex(7, "gripper_action", 0.0, 1.0),),
    )
    runner.last_target = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    runner.base = SimpleNamespace(
        arena_world=SimpleNamespace(
            get_pose_w=lambda name: poses[name],
        )
    )
    runner._decode_values = lambda values: np.asarray(values, dtype=np.float32)

    target = runner._target_from_pose_command({
        "reference": "tool",
        "offset": [1.0, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "gripper": 1,
    })

    np.testing.assert_allclose(target[:3], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(target[3:7], [0, 0, 0, 1], atol=1e-6)
    assert target[7] == 1


def test_scene_reference_target_accepts_world_orientation():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner
    from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.indices = AbsoluteCommandIndices(
        8,
        (PoseCommandIndices(slice(0, 7), "robot", "base_link"),),
        (),
        (GripperCommandIndex(7, "gripper_action", 0.0, 1.0),),
    )
    runner.last_target = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    poses = {
        "tool": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "robot": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    }
    runner.base = SimpleNamespace(arena_world=SimpleNamespace(get_pose_w=lambda name: poses[name]))
    runner._decode_values = lambda values: np.asarray(values, dtype=np.float32)

    target = runner._target_from_pose_command({
        "reference": "tool",
        "orientation": [0.0, 2**-0.5, 0.0, 2**-0.5],
        "orientation_frame": "world",
    })

    np.testing.assert_allclose(target[3:7], [2**-0.5, 0.0, -(2**-0.5), 0.0], atol=1e-6)


def test_step_target_drives_wrapped_environment():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    wrapped_env = SimpleNamespace(
        step=Mock(return_value=(None, None, torch.tensor([False]), torch.tensor([False]), {}))
    )
    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.env = wrapped_env
    runner.adapter = SimpleNamespace(command_to_action=lambda env, target: torch.tensor(target))
    runner.steps = 0
    runner.pose_results = []
    runner.viewport = None
    target = np.arange(8, dtype=np.float32)

    runner._step_target(target)

    action = wrapped_env.step.call_args.args[0]
    assert tuple(action.shape) == (1, 8)
    torch.testing.assert_close(action, torch.tensor(target).unsqueeze(0))
    np.testing.assert_array_equal(runner.last_target, target)


@pytest.mark.parametrize("step_dt", [1 / 15, 1 / 60])
def test_video_duration_tracks_simulation_time(step_dt):
    from isaaclab_arena.scripts.robot_control_runner import _VIDEO_FPS, RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.env = SimpleNamespace(step=Mock(return_value=(None, None, torch.tensor([False]), torch.tensor([False]), {})))
    runner.base = SimpleNamespace(step_dt=step_dt)
    runner.adapter = SimpleNamespace(command_to_action=lambda env, target: torch.tensor(target))
    runner.steps = 0
    runner.frames = 0
    runner.viewport = object()
    runner.writer = Mock()
    runner.last_image = np.zeros((2, 2, 3), dtype=np.uint8)

    def capture_frame():
        runner.writer.append_data(runner.last_image)
        runner.frames += 1

    runner._capture_frame = capture_frame
    for _ in range(round(2 / step_dt)):
        runner._step_target(np.zeros(8))

    assert runner.frames == 2 * _VIDEO_FPS
    assert runner.writer.append_data.call_count == runner.frames


def test_pose_holds_goal_until_motion_generator_reports_reached():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = SimpleNamespace(max_pose_steps=5)
    target = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    runner._target_from_pose_command = lambda command: target
    runner._current_target = lambda: target
    runner._step_target = Mock()
    runner.steps = 0
    runner.pose_results = []
    runner.env = object()
    runner.motion_generator = SimpleNamespace(
        compute_command=lambda goal, env, env_id: goal.copy(),
        goal_reached=Mock(side_effect=[False, True]),
        goal_blocked=Mock(return_value=False),
    )

    assert runner.pose({"action": "pose", "values": target.tolist()}) == "reached"
    assert runner._step_target.call_count == 2


def test_pose_dispatches_curobo_planner_before_holding_goal():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = SimpleNamespace(max_pose_steps=2)
    target = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    runner._target_from_pose_command = lambda command: target
    runner._current_target = lambda: target
    runner._execute_curobo_path = Mock()
    runner._step_target = Mock()
    runner.steps = 0
    runner.pose_results = []
    runner.env = object()
    runner.motion_generator = SimpleNamespace(
        compute_command=lambda goal, env, env_id: goal,
        goal_reached=lambda goal, env, env_id: True,
        goal_blocked=lambda goal, env, env_id: False,
    )
    command = {"action": "pose", "values": target.tolist(), "planner": "curobo"}

    assert runner.pose(command) == "reached"
    runner._execute_curobo_path.assert_called_once_with(target, command)


def test_pose_rejects_blocked_goal_unless_explicitly_accepted():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = SimpleNamespace(max_pose_steps=2)
    runner._target_from_pose_command = lambda command: np.zeros(8, dtype=np.float32)
    runner._current_target = lambda: np.zeros(8, dtype=np.float32)
    runner._step_target = Mock()
    runner.steps = 0
    runner.pose_results = []
    runner.env = object()
    runner.motion_generator = SimpleNamespace(
        compute_command=lambda goal, env, env_id: goal,
        goal_reached=lambda goal, env, env_id: False,
        goal_blocked=lambda goal, env, env_id: True,
    )

    with pytest.raises(AssertionError, match="blocked"):
        runner.pose({"action": "pose", "position": [0, 0, 0]})
    assert runner.pose({"action": "pose", "position": [0, 0, 0], "accept_blocked": True}) == "blocked"


def test_trajectory_applies_command_defaults_to_each_waypoint():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.pose = Mock()
    waypoints = [{"position": [0, 0, 0]}, {"position": [0, 0, 1], "max_steps": 4}]

    runner.trajectory({
        "action": "trajectory",
        "waypoints": waypoints,
        "max_steps": 10,
        "accept_blocked": True,
    })

    assert runner.pose.call_args_list[0].args[0] == {
        "position": [0, 0, 0],
        "max_steps": 10,
        "accept_blocked": True,
    }
    assert runner.pose.call_args_list[1].args[0]["max_steps"] == 4


def test_constructor_rejects_incompatible_action_layout(monkeypatch, tmp_path):
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    adapter = SimpleNamespace(is_robot_action_space_supported=lambda env: False)
    monkeypatch.setattr(
        "isaaclab_arena_vlm_agent.droid_adapter.DroidVLMAdapter.make_goal_action_adapter",
        lambda self, position_tolerance_m, rotation_tolerance_rad: adapter,
    )
    args = Namespace(
        output=tmp_path,
        position_tolerance_m=0.01,
        rotation_tolerance_rad=0.1,
        no_record=True,
    )

    with pytest.raises(AssertionError, match="droid_abs_ik"):
        RobotCommandRunner(SimpleNamespace(unwrapped=object()), args)


def test_run_recipe_closes_environment_when_execution_fails(monkeypatch):
    from isaaclab_arena.scripts import robot_control_runner

    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(enable_scene_query_support=False),
        recorders=object(),
        episode_recorders=object(),
    )

    class FakeEnvironment:
        def __init__(self):
            self.reset_count = 0
            self.unwrapped = self
            self.episode_limit = None
            self.close_count = 0

        def configure_episode_limit(self, count):
            assert self.reset_count == 0
            self.episode_limit = count

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
        def __init__(self, wrapped_env, args, embodiment):
            assert wrapped_env is env
            assert embodiment is None

        def run(self, commands):
            raise RuntimeError("recipe failed")

    monkeypatch.setattr(robot_control_runner, "RobotCommandRunner", FailingRunner)

    with pytest.raises(RuntimeError, match="recipe failed"):
        robot_control_runner._run_recipe(builder, Namespace(), ({"action": "finish"},))

    assert env_cfg.sim.enable_scene_query_support
    assert env_cfg.recorders == {}
    assert callable(env_cfg.episode_recorders.robot_control.func)
    assert env.reset_count == 1
    assert env.episode_limit is None
    assert env.close_count == 1


def test_failed_command_still_writes_diagnostics_and_closes_video():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = Namespace(settle_steps=0, allow_incomplete=False)
    runner.wait = Mock()
    runner._execute = Mock(side_effect=AssertionError("pose blocked"))
    runner._write_report = Mock(return_value={"success": [False]})
    runner.current_subtask_index = 0
    runner.current_subtask_title = "Close the lid"
    runner.mark = Mock()
    runner.writer = Mock()

    with pytest.raises(AssertionError, match="pose blocked"):
        runner.run(({"action": "pose", "position": [0.1, 0.2, 0.3]},))

    runner._write_report.assert_called_once_with("pose blocked")
    runner.mark.assert_called_once_with(0, "failure", "Close the lid")
    runner.writer.close.assert_called_once()


def test_completed_recipe_still_requires_native_task_success():
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = Namespace(settle_steps=0, allow_incomplete=False)
    runner.wait = Mock()
    runner._execute = Mock(return_value=False)
    runner._write_report = Mock(return_value={"success": [False]})
    runner.writer = None

    with pytest.raises(AssertionError, match="Final task conditions failed"):
        runner.run(({"action": "finish"},))

    runner._write_report.assert_called_once_with(None)


def test_success_stops_inside_wait_and_keeps_state_before_automatic_reset(tmp_path):
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.args = Namespace(settle_steps=0, allow_incomplete=False)
    runner.output = tmp_path
    runner.steps = 0
    runner.pose_results = []
    runner._success_report = None
    runner.viewport = None
    runner.writer = Mock()
    runner.current_subtask_index = 0
    runner.current_subtask_title = "Close the lid"
    runner.mark = Mock()
    runner.last_target = np.zeros(8)
    runner._current_target = lambda: runner.last_target
    runner.adapter = SimpleNamespace(command_to_action=lambda env, target: torch.tensor(target))
    state = {"success": [False], "episode_index": 0}
    runner._report_state = Mock(side_effect=lambda: dict(state))
    runner.base = SimpleNamespace(
        termination_manager=SimpleNamespace(get_term=lambda name: torch.tensor([True])),
    )

    def step(action):
        state["success"] = [True]
        # The episode recorder observes success before same-step autoreset.
        runner.record_success_before_reset()
        state.update(success=[False], episode_index=1)
        return None, None, torch.tensor([True]), torch.tensor([False]), {}

    runner.env = SimpleNamespace(step=Mock(side_effect=step))
    runner.run(({"action": "wait", "steps": 100}, {"action": "wait", "steps": 100}))

    runner.env.step.assert_called_once()
    assert runner.steps == 1
    assert state["episode_index"] == 1  # Arena was allowed to reset normally.
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["success"] == [True]
    assert report["episode_index"] == 0
    assert report["stopped_on_success"]
    assert report["execution_error"] is None
    runner._report_state.assert_called_once()
    runner.mark.assert_called_once_with(0, "end", "Close the lid")
    runner.writer.close.assert_called_once()


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_failure_or_timeout_is_not_task_success(terminated, truncated):
    from isaaclab_arena.scripts.robot_control_runner import RobotCommandRunner

    runner = RobotCommandRunner.__new__(RobotCommandRunner)
    runner.env = SimpleNamespace(
        step=lambda action: (None, None, torch.tensor([terminated]), torch.tensor([truncated]), {})
    )
    runner.base = SimpleNamespace(
        termination_manager=SimpleNamespace(get_term=lambda name: torch.tensor([False])),
    )
    runner.adapter = SimpleNamespace(command_to_action=lambda env, target: torch.tensor(target))
    runner.steps = 0
    runner.viewport = None

    runner._step_target(np.zeros(8))
    assert runner.steps == 1
