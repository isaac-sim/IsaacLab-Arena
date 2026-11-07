# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest


def test_load_force_control_recipe_separates_setup_from_runtime_commands(tmp_path):
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_recipe import (
        load_force_control_recipe,
    )

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


def test_replace_existing_actuators_keeps_all_recipe_joint_setups():
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        _apply_joint_setups,
    )

    cabinet_cfg = SimpleNamespace(actuators={"original": object()})
    env_cfg = SimpleNamespace(scene=SimpleNamespace(cabinet=cabinet_cfg))
    setups = (
        {"name": "cabinet", "joint_name": "door", "replace_existing": True},
        {"name": "cabinet", "joint_name": "drawer", "replace_existing": True},
    )

    _apply_joint_setups(env_cfg, setups)

    assert set(cabinet_cfg.actuators) == {"force_control_door", "force_control_drawer"}


def test_reset_joint_hold_is_active_before_initial_settling():
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
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
    runner.video = None
    observed = []
    runner._set_joint_hold = lambda *args: observed.append(args)
    runner.step = lambda count: observed.append(count)
    runner._execute_commands = lambda commands: None
    runner._write_report = lambda mode: {"success": [True]}
    setup = {
        "action": "setup_joint",
        "name": "scope",
        "joint_name": "button",
        "settle_hold": {"gain": 2.0, "damping": 0.015, "limit": 0.01},
    }

    runner.run((), (setup,))

    assert observed[0][:2] == ("scope", "button")
    assert observed[0][2:] == pytest.approx((-0.001, 2.0, 0.015, 0.01))
    assert observed[1] == 240


def test_combine_reference_goal_uses_only_selected_reference_axes():
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        _combine_reference_goal,
    )

    result = _combine_reference_goal(
        goal=[1.0, 2.0, 0.56],
        reference=[0.2, -0.1, 0.8],
        offset=[0.05, 0.06, 0.0],
        reference_axes=["x", "y"],
    )

    assert result == pytest.approx([0.25, -0.04, 0.56])


def test_check_timeout_writes_failure_snapshot(tmp_path):
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.output = tmp_path
    runner.env = SimpleNamespace(_sim_step_counter=10, cfg=SimpleNamespace(decimation=2))
    runner.start_sim_step = 0
    runner.video = None
    runner._task_completion = lambda: torch.tensor([[False]])
    runner._criterion_status = lambda: {"criterion": [[False]]}
    runner._collect_state = lambda: {"completion": [[False]], "current_predicates": {"criterion": [[False]]}}
    runner.step = lambda: setattr(runner.env, "_sim_step_counter", runner.steps + 2)

    with pytest.raises(AssertionError, match="diagnostics:"):
        runner.check(0, timeout_steps=2)

    [snapshot_path] = tmp_path.glob("failure_check_timeout_subtask_00_step_*.json")
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["trigger"] == "check_timeout"
    assert snapshot["completion"] == [[False]]


def test_overlay_progress_banner_draws_title_and_preserves_scene_pixels():
    import numpy as np

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_video import (
        overlay_progress_banner,
    )

    progress = {"index": 1, "title": "Put the second banana in the bowl", "completion": [True, False]}
    frame = np.full((120, 320, 3), 200, dtype=np.uint8)

    annotated = overlay_progress_banner(frame, progress)

    assert not np.array_equal(annotated[:76], frame[:76])
    assert np.array_equal(annotated[76:], frame[76:])


def test_run_recipe_closes_environment_when_execution_fails(monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation import force_demo_runner
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_recipe import (
        ForceControlRecipe,
    )

    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(enable_scene_query_support=False),
        recorders=object(),
        episode_recorders=object(),
    )

    class FakeEnvironment:
        def __init__(self):
            self.unwrapped = SimpleNamespace(configure_episode_limit=lambda count: None)
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
            pass

        def run(self, commands, joint_setups, report_mode):
            raise RuntimeError("recipe failed")

    monkeypatch.setattr(force_demo_runner, "PhysicalCommandRunner", FailingRunner)
    monkeypatch.setattr(force_demo_runner, "configure_physical_command_runner", lambda cfg, setups: None)
    recipe = ForceControlRecipe((), ({"action": "finish"},))

    with pytest.raises(RuntimeError, match="recipe failed"):
        force_demo_runner._run_recipe(builder, Namespace(no_record=True, follow_commands=False), recipe)

    assert env.reset_count == 1
    assert env.close_count == 1


def test_force_steps_delegate_to_wrapped_environment():
    from unittest.mock import Mock

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(cfg=SimpleNamespace(decimation=4), _sim_step_counter=0)
    runner.start_sim_step = 0
    runner.wrapped_env = SimpleNamespace(
        step=Mock(side_effect=lambda _: setattr(runner.env, "_sim_step_counter", runner.steps + 4))
    )
    runner.actions = object()
    runner._record_progress = Mock()
    runner.step(9)
    assert runner.wrapped_env.step.call_count == 3
    runner.wrapped_env.step.assert_called_with(runner.actions)
    assert runner.steps == 12


@pytest.mark.parametrize("timeout_steps,initially_complete", [(0, True), (8, False)])
def test_check_accepts_completion_at_timeout_boundary(timeout_steps, initially_complete):
    import torch

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(cfg=SimpleNamespace(decimation=4))
    calls = []
    runner._task_completion = lambda: torch.tensor([[initially_complete or len(calls) == 2]])
    runner.step = lambda: calls.append(True)
    runner.check(0, timeout_steps=timeout_steps)
    assert len(calls) == (0 if initially_complete else 2)


def test_diagnostics_use_initialized_named_and_temporal_predicates():
    import torch

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    def predicate(env, value):
        return torch.tensor([value])

    term = TerminationTermCfg(func=predicate, params={"value": True})
    criteria = [
        CompletionCriteria(
            name="named",
            predicate_sequences={"first": [term], "second": [TrueForConsecutiveStepsCfg(term, required_steps=3)]},
        ),
        CompletionCriteria(
            name="single", predicate_sequence=[TerminationTermCfg(func=predicate, params={"value": False})]
        ),
    ]
    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(num_envs=1, device="cpu")
    runner.env.progress_tracker = ProgressTracker(criteria, 1, "cpu", env=runner.env)
    before = runner.env.progress_tracker.get_consecutive_step_summary(0)
    assert runner._criterion_status() == {"named": {"first": [[True]], "second": [[True]]}, "single": [[False]]}
    assert runner.env.progress_tracker.get_consecutive_step_summary(0) == before


def test_follow_waits_for_complete_records_and_skips_blank_lines(tmp_path, monkeypatch):
    from isaaclab_arena_examples.agentic_environment_generation import force_demo_runner as module
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_recipe import (
        load_force_control_recipe,
    )

    path = tmp_path / "follow.jsonl"
    path.write_text('\n{"action":"wait",')
    assert load_force_control_recipe(path, require_runtime_commands=False).commands == ()

    def append_rest(_):
        with path.open("a") as stream:
            stream.write('"steps":0}\n  \n{"action":"finish"}\n')

    monkeypatch.setattr(module.time, "sleep", append_rest)
    commands = list(module._follow_commands(path, lambda: None))
    assert commands == [{"action": "wait", "steps": 0}, {"action": "finish"}]


def test_follow_buffers_an_incomplete_utf8_character(tmp_path):
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_recipe import (
        load_force_control_recipe,
    )

    path = tmp_path / "follow.jsonl"
    line = '{"action":"mark","index":0,"boundary":"start","title":"90°"}\n'.encode()
    split = line.index("°".encode()) + 1
    path.write_bytes(line[:split])
    assert load_force_control_recipe(path, require_runtime_commands=False).commands == ()
    with path.open("ab") as stream:
        stream.write(line[split:])
    assert load_force_control_recipe(path, require_runtime_commands=False).commands[0]["title"] == "90°"


@pytest.mark.parametrize("physics_dt,decimation", [(1 / 120, 8), (1 / 200, 4)])
def test_lab_recording_preserves_simulation_time(tmp_path, physics_dt, decimation):
    import numpy as np

    from isaaclab.envs.utils.video_recorder import VideoRecorder

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_video import (
        ForceControlVideo,
        configure_video,
    )

    cfg = SimpleNamespace(sim=SimpleNamespace(dt=physics_dt), decimation=decimation)
    configure_video(cfg, tmp_path)
    env = SimpleNamespace(
        step_dt=physics_dt * decimation,
        sim=SimpleNamespace(visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type="kit"))]),
    )
    recorder = VideoRecorder(cfg.video_recorders[0], env)
    env.video_recorders = [recorder]
    recorder._get_frame = lambda: np.full((96, 128, 3), 120, dtype=np.uint8)
    video = ForceControlVideo(env, tmp_path)
    video.record_progress(0, {"index": 0, "title": "Test", "completion": [False]})
    for _ in range(round(1 / env.step_dt)):
        recorder.step()
    video.close()
    import imageio.v2 as imageio

    with imageio.get_reader(str(tmp_path / "demo.mp4")) as reader:
        assert reader.get_meta_data()["duration"] == pytest.approx(1.0, abs=0.01)
    assert video.frames == round(1 / env.step_dt)
    assert (tmp_path / "video_progress.json").exists()


def test_failed_recipe_flushes_lab_video_and_diagnostics(tmp_path):
    import numpy as np
    import torch

    from isaaclab.envs.utils.video_recorder import VideoRecorder

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.force_control_video import (
        ForceControlVideo,
        configure_video,
    )
    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    frame = np.full((96, 128, 3), 120, dtype=np.uint8)
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.005), decimation=4)
    configure_video(cfg, tmp_path)
    env = SimpleNamespace(
        step_dt=0.02,
        physics_dt=0.005,
        cfg=cfg,
        _sim_step_counter=0,
        sim=SimpleNamespace(
            render=lambda: None,
            visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type="kit"), render_rgb_array=lambda: frame)],
        ),
    )
    recorder = VideoRecorder(cfg.video_recorders[0], env)
    recorder._get_frame = lambda: frame
    env.video_recorders = [recorder]
    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = env
    runner.start_sim_step = 0
    runner.args = Namespace(follow_commands=False, allow_incomplete=False)
    runner.output = tmp_path
    runner.video = ForceControlVideo(env, tmp_path)
    runner._task_completion = lambda: torch.tensor([[False]])
    runner.current_subtask_index = 0
    runner.current_subtask_title = "Expected failure"
    runner._criterion_status = lambda: {"criterion": [[False]]}
    diagnostic = {
        "completion": [[False]],
        "current_predicates": {"criterion": [[False]]},
        "contact_sensors": {"cube": {"net_forces_w": [[[0, 0, 9.81]]]}},
        "wrenches": {"cube": {"force": [[0, 0, 9.81]]}},
    }
    runner._collect_state = lambda: dict(diagnostic)

    def advance():
        env._sim_step_counter += cfg.decimation
        recorder.step()
        runner._record_progress()

    runner._step_env = advance
    runner._record_progress()
    with pytest.raises(AssertionError, match="Subtask 0 did not complete"):
        runner.run(({"action": "check", "index": 0, "timeout_steps": 0},))
    [snapshot] = tmp_path.glob("failure_check_timeout_*.json")
    state = json.loads(snapshot.read_text())
    assert state["contact_sensors"] == diagnostic["contact_sensors"]
    assert state["wrenches"] == diagnostic["wrenches"]
    assert (tmp_path / state["viewport"]).exists()
    assert (tmp_path / "demo.mp4").exists()
    assert runner.video.frames == 60


def test_execution_error_keeps_json_when_video_capture_and_cleanup_fail(tmp_path):
    from unittest.mock import Mock

    from isaaclab_arena_examples.agentic_environment_generation.force_demo.physical_command_runner import (
        PhysicalCommandRunner,
    )

    runner = PhysicalCommandRunner.__new__(PhysicalCommandRunner)
    runner.env = SimpleNamespace(_sim_step_counter=0)
    runner.start_sim_step = 0
    runner.output = tmp_path
    runner.step = Mock(side_effect=RuntimeError("physics failed"))
    runner.video = SimpleNamespace(
        snapshot=Mock(side_effect=RuntimeError("capture failed")),
        close=Mock(side_effect=RuntimeError("encoding failed")),
    )
    runner._collect_state = lambda: {"poses": {"cube": [[0, 0, 1, 0, 0, 0, 1]]}}
    runner._progress = lambda: {}
    with pytest.raises(RuntimeError, match="^physics failed$"):
        runner.run(())
    [snapshot] = tmp_path.glob("failure_execution_error_*.json")
    assert json.loads(snapshot.read_text())["poses"]["cube"][0][2] == 1
