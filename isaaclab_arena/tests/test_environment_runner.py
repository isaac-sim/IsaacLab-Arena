# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import io
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.evaluation import environment_runner
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _interactive_runner_args(**overrides) -> argparse.Namespace:
    argument_values = {
        "headless": False,
        "visualizer": ["kit"],
        "num_envs": 1,
        "distributed": False,
        "presets": None,
        "list_variations": False,
        "device": "cpu",
    }
    argument_values.update(overrides)
    return argparse.Namespace(**argument_values)


class _FakeEnvironment:
    def __init__(
        self,
        *,
        action_space_shape: tuple[int, ...] = (1, 2),
        num_envs: int = 1,
        termination_terms: dict[str, torch.Tensor] | None = None,
    ) -> None:
        termination_terms = termination_terms or {}
        termination_manager = SimpleNamespace(
            active_terms=list(termination_terms),
            get_term=lambda term_name: termination_terms[term_name],
        )
        self.unwrapped = SimpleNamespace(
            device="cpu",
            num_envs=num_envs,
            step_dt=0.02,
            termination_manager=termination_manager,
        )
        self.action_space = SimpleNamespace(shape=action_space_shape)
        self.reset_count = 0
        self.step_actions: list[torch.Tensor] = []
        self.close_count = 0

    def reset(self):
        self.reset_count += 1
        return {}, {}

    def step(self, actions: torch.Tensor):
        self.step_actions.append(actions.clone())
        terminated = torch.tensor([True])
        truncated = torch.tensor([False])
        return {}, None, terminated, truncated, {}

    def close(self) -> None:
        self.close_count += 1


def test_use_kit_visualizer_by_default_selects_kit_only_when_visualizer_was_omitted():
    default_args = argparse.Namespace(visualizer=None, visualizer_explicit=False, headless=False)
    environment_runner.use_kit_visualizer_by_default(default_args)
    assert default_args.visualizer == ["kit"]

    explicit_none_args = argparse.Namespace(visualizer=None, visualizer_explicit=True, headless=False)
    environment_runner.use_kit_visualizer_by_default(explicit_none_args)
    assert explicit_none_args.visualizer is None

    headless_args = argparse.Namespace(visualizer=None, visualizer_explicit=False, headless=True)
    environment_runner.use_kit_visualizer_by_default(headless_args)
    assert headless_args.visualizer is None


def test_assert_interactive_runner_args_accepts_one_physx_kit_environment():
    environment_runner.assert_interactive_runner_args(_interactive_runner_args())


def test_assert_gui_environment_rejects_headless_environment_variable(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    with pytest.raises(AssertionError, match="unset HEADLESS"):
        environment_runner.assert_gui_environment()

    monkeypatch.setenv("HEADLESS", "0")
    environment_runner.assert_gui_environment()


@pytest.mark.parametrize(
    ("argument_overrides", "expected_message"),
    [
        ({"headless": True}, "requires the Kit GUI"),
        ({"visualizer": None}, "requires the Kit GUI"),
        ({"visualizer": ["viser"]}, "requires the Kit GUI"),
        ({"num_envs": 2}, "exactly one environment"),
        ({"distributed": True}, "does not support distributed execution"),
        ({"presets": "newton"}, "requires PhysX"),
        ({"list_variations": True}, "does not support --list_variations"),
        ({"device": "cuda:0"}, "requires CPU PhysX"),
    ],
)
def test_assert_interactive_runner_args_rejects_unsupported_configuration(argument_overrides, expected_message):
    with pytest.raises(AssertionError, match=expected_message):
        environment_runner.assert_interactive_runner_args(_interactive_runner_args(**argument_overrides))


def test_disable_timeout_terminations_removes_canonical_and_flagged_timeout_terms():
    canonical_timeout = object()
    custom_timeout = SimpleNamespace(time_out=True)
    success = SimpleNamespace(time_out=False)
    terminations_cfg = SimpleNamespace(
        time_out=canonical_timeout,
        connection_lost=custom_timeout,
        success=success,
    )
    env_cfg = SimpleNamespace(terminations=terminations_cfg)

    disabled_term_names = environment_runner.disable_timeout_terminations(env_cfg)

    assert disabled_term_names == ["time_out", "connection_lost"]
    assert terminations_cfg.time_out is None
    assert terminations_cfg.connection_lost is None
    assert terminations_cfg.success is success


def test_disable_timeout_terminations_handles_environment_without_terminations():
    assert environment_runner.disable_timeout_terminations(SimpleNamespace(terminations=None)) == []


def test_get_idle_actions_returns_zeros_matching_the_environment_action_space():
    env = _FakeEnvironment(action_space_shape=(1, 3))

    actions = environment_runner.get_idle_actions(env, SimpleNamespace())

    assert actions.device.type == "cpu"
    assert torch.equal(actions, torch.zeros((1, 3)))


def test_get_idle_actions_repeats_the_configured_action_for_each_environment():
    env = _FakeEnvironment(action_space_shape=(2, 2), num_envs=2)
    configured_idle_action = torch.tensor([[0.25, -0.5]])
    env_cfg = SimpleNamespace(idle_action=configured_idle_action)

    actions = environment_runner.get_idle_actions(env, env_cfg)

    assert torch.equal(actions, torch.tensor([[0.25, -0.5], [0.25, -0.5]]))


def test_get_fired_termination_terms_returns_every_fired_term_and_environment_id():
    env = _FakeEnvironment(
        num_envs=3,
        action_space_shape=(3, 2),
        termination_terms={
            "success": torch.tensor([True, False, True]),
            "robot_fell": torch.tensor([False, True, False]),
            "inactive": torch.tensor([False, False, False]),
        },
    )

    assert environment_runner.get_fired_termination_terms(env) == {
        "success": [0, 2],
        "robot_fell": [1],
    }


def test_log_fired_termination_terms_prints_all_named_reset_causes(capsys):
    env = _FakeEnvironment(
        termination_terms={
            "success": torch.tensor([True]),
            "robot_fell": torch.tensor([True]),
            "inactive": torch.tensor([False]),
        }
    )

    environment_runner.log_fired_termination_terms(
        env,
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
    )

    output = capsys.readouterr().out
    assert "Termination 'success' fired for env IDs [0]; environment reset." in output
    assert "Termination 'robot_fell' fired for env IDs [0]; environment reset." in output
    assert "inactive" not in output


def test_log_fired_termination_terms_does_not_inspect_terms_without_a_reset(capsys):
    class UnexpectedTerminationManagerAccess:
        @property
        def active_terms(self):
            raise AssertionError("Termination terms should not be inspected without a reset")

    env = _FakeEnvironment()
    env.unwrapped.termination_manager = UnexpectedTerminationManagerAccess()

    environment_runner.log_fired_termination_terms(
        env,
        terminated=torch.tensor([False]),
        truncated=torch.tensor([False]),
    )

    assert capsys.readouterr().out == ""


def test_run_environment_resets_and_steps_once_before_the_application_stops(capsys):
    class OneIterationSimulationApp:
        def __init__(self) -> None:
            self.running_checks = 0

        def is_running(self) -> bool:
            self.running_checks += 1
            return self.running_checks == 1

        def is_exiting(self) -> bool:
            return False

    class CountingRateLimiter:
        def __init__(self) -> None:
            self.sleep_count = 0

        def sleep(self) -> None:
            self.sleep_count += 1

    env = _FakeEnvironment(termination_terms={"success": torch.tensor([True])})
    rate_limiter = CountingRateLimiter()

    environment_runner.run_environment(
        OneIterationSimulationApp(),
        env,
        SimpleNamespace(),
        rate_limiter=rate_limiter,
    )

    assert env.reset_count == 1
    assert len(env.step_actions) == 1
    assert torch.equal(env.step_actions[0], torch.zeros((1, 2)))
    assert rate_limiter.sleep_count == 1
    assert "Termination 'success' fired" in capsys.readouterr().out


def _test_success_term_is_logged_after_automatic_reset(simulation_app) -> bool:
    from isaaclab_arena.tests.test_open_door import get_test_environment

    env, microwave = get_test_environment(remove_reset_door_state_event=False, num_envs=1)
    try:
        microwave.open(env, env_ids=None)
        actions = torch.zeros(env.action_space.shape, device=env.device)
        _, _, terminated, truncated, _ = env.step(actions)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            environment_runner.log_fired_termination_terms(env, terminated, truncated)

        assert terminated.item()
        assert "Termination 'success' fired" in output.getvalue()
        assert not microwave.is_open(env).item()
    finally:
        env.close()
    return True


def test_success_term_is_logged_after_automatic_reset():
    result = run_simulation_app_function(_test_success_term_is_logged_after_automatic_reset, headless=True)
    assert result


def _test_mouse_interaction_uses_d6_grab_for_current_stage(simulation_app) -> bool:
    import carb
    import omni.kit.app
    import omni.physx.bindings._physx as physx_bindings
    import omni.usd

    environment_runner.enable_mouse_interaction()

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    assert extension_manager.is_extension_enabled("omni.physx.ui")

    settings = carb.settings.get_settings()
    expected_settings = {
        physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED: True,
        physx_bindings.SETTING_MOUSE_GRAB: True,
        physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE: False,
        physx_bindings.SETTING_MOUSE_GRAB_IGNORE_INVISBLE: False,
    }
    physics_settings = omni.usd.get_context().get_stage().GetRootLayer().customLayerData["physicsSettings"]
    for setting_path, expected_value in expected_settings.items():
        assert settings.get_as_bool(setting_path) is expected_value
        assert physics_settings[setting_path] is expected_value
    return True


def test_mouse_interaction_uses_d6_grab_for_current_stage():
    result = run_simulation_app_function(_test_mouse_interaction_uses_d6_grab_for_current_stage, headless=True)
    assert result


def test_main_closes_the_environment_when_the_run_loop_fails(monkeypatch):
    app_args = _interactive_runner_args(visualizer=None, visualizer_explicit=False, device="cuda:0")
    full_args = _interactive_runner_args(visualizer=None, visualizer_explicit=False, device="cuda:0")

    class SequencedArgumentParser:
        def __init__(self) -> None:
            self.allow_abbrev = True
            self._parse_results = [(app_args, ["gr1_open_microwave"]), (full_args, [])]

        def set_defaults(self, **defaults) -> None:
            for parsed_args, _ in self._parse_results:
                for argument_name, default_value in defaults.items():
                    setattr(parsed_args, argument_name, default_value)

        def parse_known_args(self):
            return self._parse_results.pop(0)

    class FakeSimulationAppContext:
        def __init__(self, received_app_args) -> None:
            self.received_app_args = received_app_args
            assert received_app_args.device == "cpu"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    lifecycle_events = []
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(enable_scene_query_support=False),
        terminations=None,
        recorders=object(),
        episode_recorders=object(),
    )
    env = _FakeEnvironment()

    class FakeArenaBuilder:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(device=full_args.device)

        def compose_manager_cfg(self):
            return env_cfg, {"example_kwarg": "value"}

        def make_registered(self, received_env_cfg, received_env_kwargs):
            lifecycle_events.append("make_environment")
            assert self.cfg.device == "cpu"
            assert received_env_cfg is env_cfg
            assert received_env_cfg.sim.enable_scene_query_support
            assert received_env_cfg.recorders == {}
            assert received_env_cfg.episode_recorders == {}
            assert received_env_kwargs == {"example_kwarg": "value"}
            return env

    parser = SequencedArgumentParser()
    monkeypatch.setattr(environment_runner, "get_isaaclab_arena_cli_parser", lambda: parser)
    monkeypatch.setattr(
        environment_runner,
        "get_isaaclab_arena_environments_cli_parser",
        lambda received_parser: received_parser,
    )
    monkeypatch.setattr(environment_runner, "SimulationAppContext", FakeSimulationAppContext)
    monkeypatch.setattr(environment_runner, "assert_hydra_overrides", lambda overrides, received_parser: None)
    monkeypatch.setattr(environment_runner, "assert_gui_environment", lambda: None)
    monkeypatch.setattr(
        environment_runner,
        "get_arena_builder_from_cli",
        lambda args_cli, hydra_overrides: FakeArenaBuilder(),
    )
    monkeypatch.setattr(
        environment_runner,
        "enable_mouse_interaction",
        lambda: lifecycle_events.append("enable_mouse_interaction"),
    )

    def fail_during_environment_run(simulation_app, received_env, received_env_cfg):
        lifecycle_events.append("run_environment")
        raise RuntimeError("run failed")

    monkeypatch.setattr(environment_runner, "run_environment", fail_during_environment_run)

    with pytest.raises(RuntimeError, match="run failed"):
        environment_runner.main()

    assert parser.allow_abbrev is False
    assert app_args.visualizer == ["kit"]
    assert full_args.visualizer == ["kit"]
    assert app_args.device == "cpu"
    assert full_args.device == "cpu"
    assert lifecycle_events == ["make_environment", "enable_mouse_interaction", "run_environment"]
    assert env.close_count == 1
