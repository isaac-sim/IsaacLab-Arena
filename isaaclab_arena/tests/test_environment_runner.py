# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.scripts import environment_runner
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
        "disable_fabric": False,
    }
    argument_values.update(overrides)
    return argparse.Namespace(**argument_values)


class _FakeEnvironment:
    def __init__(
        self,
        *,
        action_space_shape: tuple[int, ...] = (1, 2),
    ) -> None:
        self.unwrapped = SimpleNamespace(
            device="cpu",
            step_dt=0.02,
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


def test_assert_interactive_runner_args_accepts_one_physx_kit_environment():
    environment_runner._assert_interactive_runner_args(_interactive_runner_args())


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
        environment_runner._assert_interactive_runner_args(_interactive_runner_args(**argument_overrides))


def test_run_environment_resets_and_steps_once_before_the_application_stops(monkeypatch):
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

    env = _FakeEnvironment()
    rate_limiter = CountingRateLimiter()

    def create_rate_limiter(period_seconds):
        assert period_seconds == env.unwrapped.step_dt
        return rate_limiter

    monkeypatch.setattr(environment_runner, "RateLimiter", create_rate_limiter)

    environment_runner.run_environment(
        OneIterationSimulationApp(),
        env,
    )

    assert env.reset_count == 1
    assert len(env.step_actions) == 1
    assert torch.equal(env.step_actions[0], torch.zeros((1, 2)))
    assert rate_limiter.sleep_count == 1


def _test_mouse_interaction_uses_d6_grab_for_current_stage(simulation_app) -> bool:
    import carb
    import omni.kit.app
    import omni.physx.bindings._physx as physx_bindings
    import omni.usd

    environment_runner._enable_mouse_interaction()

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
    args_cli = _interactive_runner_args(
        visualizer=None,
        visualizer_explicit=False,
        device="cuda:0",
        example_environment="gr1_open_microwave",
    )

    class FakeArgumentParser:
        def __init__(self) -> None:
            self.allow_abbrev = True

        def set_defaults(self, **defaults) -> None:
            for argument_name, default_value in defaults.items():
                setattr(args_cli, argument_name, default_value)

        def parse_known_args(self):
            return args_cli, []

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
        recorders=object(),
        episode_recorders=object(),
    )
    env = _FakeEnvironment()

    class FakeArenaBuilder:
        def __init__(self, received_args) -> None:
            self.cfg = SimpleNamespace(device=received_args.device)

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

    parser = FakeArgumentParser()
    monkeypatch.setattr(environment_runner, "get_isaaclab_arena_cli_parser", lambda: parser)
    monkeypatch.setattr(
        environment_runner,
        "get_isaaclab_arena_environments_cli_parser",
        lambda received_parser: received_parser,
    )
    monkeypatch.setattr(environment_runner, "SimulationAppContext", FakeSimulationAppContext)
    monkeypatch.setattr(environment_runner, "assert_hydra_overrides", lambda overrides, received_parser: None)
    monkeypatch.setattr(
        environment_runner,
        "get_arena_builder_from_cli",
        lambda args_cli, hydra_overrides: FakeArenaBuilder(args_cli),
    )
    monkeypatch.setattr(
        environment_runner,
        "_enable_mouse_interaction",
        lambda: lifecycle_events.append("enable_mouse_interaction"),
    )

    def fail_during_environment_run(simulation_app, received_env):
        lifecycle_events.append("run_environment")
        raise RuntimeError("run failed")

    monkeypatch.setattr(environment_runner, "run_environment", fail_during_environment_run)

    with pytest.raises(RuntimeError, match="run failed"):
        environment_runner.main()

    assert parser.allow_abbrev is False
    assert args_cli.visualizer == ["kit"]
    assert args_cli.device == "cpu"
    assert args_cli.disable_fabric
    assert args_cli.example_environment == "gr1_open_microwave"
    assert lifecycle_events == ["make_environment", "enable_mouse_interaction", "run_environment"]
    assert env.close_count == 1
