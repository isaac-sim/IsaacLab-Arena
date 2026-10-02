# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check condition sampling, replay, and partial resets without simulation."""

import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.variations.sampling_context import VariationReplay, VariationSamplingContext
from isaaclab_arena.variations.variation_recorder import VariationRecorder
from isaaclab_arena_environments.return_to_service.scenarios import SCENARIOS
from isaaclab_arena_environments.return_to_service.variations import (
    ServiceScenarioVariation,
    ServiceScenarioVariationCfg,
)


def _env():
    return SimpleNamespace(
        num_envs=3,
        device="cpu",
        task_runtime=SimpleNamespace(scenario_names=["healthy", "filter", "combined"]),
        get_episode_index=lambda env_id: env_id + 1,
    )


def _apply(variation, env, ids):
    _, event = variation.build_event_cfg()
    event.func(env, ids, **event.params)


def test_scenario_variation_is_opt_in_and_preserves_public_work_order():
    variation = ServiceScenarioVariation()
    assert not variation.enabled
    assert variation.cfg.scenario_names == list(SCENARIOS)
    assert len({SCENARIOS[name].work_order for name in variation.cfg.scenario_names}) == 1


def test_partial_reset_updates_only_selected_scenarios_and_records_choices():
    env = _env()
    variation = ServiceScenarioVariation(ServiceScenarioVariationCfg(enabled=True, scenario_names=["battery"]))
    context = VariationSamplingContext(seed=7)
    context.bind_env(env)
    variation.bind_sampling_context(context, "body.scenario")
    recorder = VariationRecorder()
    recorder.attach({"body": [variation]})
    recorder.bind_env(env)
    _apply(variation, env, torch.tensor([2, 0]))
    assert env.task_runtime.scenario_names == ["battery", "filter", "battery"]
    assert recorder["body.scenario"].sample_for_episode(2, 3) == "battery"
    assert recorder["body.scenario"].sample_for_episode(0, 1) == "battery"
    assert recorder["body.scenario"].sample_for_episode(1, 2) is None


def test_scenario_replay_applies_realized_conditions_without_resampling():
    env = _env()
    rows = [
        {"env_id": 0, "episode_in_env": 1, "variations": {"body.scenario": "obstruction"}},
        {"env_id": 2, "episode_in_env": 3, "variations": {"body.scenario": "battery_filter"}},
    ]
    context = VariationSamplingContext(replay=VariationReplay(rows))
    context.bind_env(env)
    variation = ServiceScenarioVariation()
    variation.bind_sampling_context(context, "body.scenario")
    _apply(variation, env, torch.tensor([2, 0]))
    assert env.task_runtime.scenario_names == ["obstruction", "filter", "battery_filter"]


@pytest.mark.parametrize("names", [[], ["unknown"], ["healthy", "healthy"]])
def test_invalid_scenario_domains_are_rejected(names):
    with pytest.raises(AssertionError):
        ServiceScenarioVariation(ServiceScenarioVariationCfg(scenario_names=names)).validate_cfg()


def test_invalid_replay_does_not_change_any_runtime_condition():
    env = _env()
    before = list(env.task_runtime.scenario_names)
    replay = VariationReplay([
        {"env_id": 0, "episode_in_env": 1, "variations": {"body.scenario": "healthy"}},
        {"env_id": 2, "episode_in_env": 3, "variations": {"body.scenario": "invented"}},
    ])
    context = VariationSamplingContext(replay=replay)
    context.bind_env(env)
    variation = ServiceScenarioVariation()
    variation.bind_sampling_context(context, "body.scenario")
    with pytest.raises(AssertionError, match="configured choices"):
        _apply(variation, env, torch.tensor([0, 2]))
    assert env.task_runtime.scenario_names == before


def test_service_variation_experiment_loads_with_typed_environment_metadata():
    from pathlib import Path

    from isaaclab_arena.hydra.typed_experiment_loader import load_arena_experiment_from_yaml
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
    from isaaclab_arena_environments.return_to_service_environment import ReturnToServiceEnvironmentCfg

    path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_arena_environments/return_to_service/experiment_configs/variations.yaml"
    )
    experiment = load_arena_experiment_from_yaml(
        path,
        environment_cfg_types={"return_to_service": ReturnToServiceEnvironmentCfg},
        policy_cfg_type_resolver=lambda name: {"zero_action": ZeroActionPolicyCfg}[name],
    )
    assert list(experiment.runs) == [
        "sampled_conditions",
        "battery_mass",
        "camera_translation",
        "translated_left",
        "rotated_right",
        "lighting",
    ]
    for name, run in experiment.runs.items():
        assert isinstance(run.environment, ReturnToServiceEnvironmentCfg)
        expected_layout = name if name in {"translated_left", "rotated_right"} else "baseline"
        assert run.environment.layout_name == expected_layout
        assert run.environment_builder.variation_seed == 73


def test_service_environment_metadata_preserves_cli_configuration():
    import argparse

    from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli
    from isaaclab_arena_environments.return_to_service_environment import ReturnToServiceEnvironmentCfg

    parser = argparse.ArgumentParser(exit_on_error=False)
    add_dataclass_cli_args(parser, ReturnToServiceEnvironmentCfg)
    args = parser.parse_args(["--scenarios", "healthy", "combined", "--layout_name", "translated_left"])
    cfg = dataclass_from_cli(ReturnToServiceEnvironmentCfg, args)
    assert cfg.scenarios == ["healthy", "combined"]
    assert cfg.layout_name == "translated_left"
