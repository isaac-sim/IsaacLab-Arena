# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the argparse compatibility path that builds graph-spec environments for Runs."""

from pathlib import Path
from types import SimpleNamespace

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation import legacy_graph_environment_cli
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, RolloutLimitCfg
from isaaclab_arena.evaluation.legacy_environment_cli_args import legacy_environment_args_to_cli_args
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.variations.variations_hydra import overrides_from_dict


def test_legacy_environment_arguments_keep_cli_order_and_boolean_flags():
    args = legacy_environment_args_to_cli_args({
        "environment": "test_env",
        "object": "box",
        "num_envs": 4,
        "mimic": True,
        "enable_cameras": False,
    })

    assert args[:3] == ["--num_envs", "4", "test_env"]
    assert args[args.index("--object") + 1] == "box"
    assert "--mimic" in args
    assert "--enable_cameras" not in args


def test_legacy_graph_builder_keeps_namespace_inside_graph_compatibility(monkeypatch):
    graph_path = Path(TestConstants.test_data_dir) / "pick_and_place_maple_table_env_graph.yaml"
    run = ArenaRunCfg(
        name="graph_environment",
        environment=LegacyGraphEnvironmentCfg(env_spec_path=str(graph_path), per_run_overrides={"num_envs": 2}),
        environment_builder=ArenaEnvBuilderCfg(num_envs=2, device="cuda:1"),
        policy=ZeroActionPolicyCfg(),
        rollout_limit=RolloutLimitCfg(num_steps=2),
        variations={"light": {"intensity": {"enabled": True}}},
    )
    parsed_args = SimpleNamespace(env_spec=str(graph_path))
    expected_arena_env = object()
    expected_builder = object()
    captured = {}

    class _Parser:
        def parse_args(self, arguments):
            captured["arguments"] = arguments
            return parsed_args

    monkeypatch.setattr(legacy_graph_environment_cli, "get_isaaclab_arena_environments_cli_parser", lambda: _Parser())

    def get_arena_env(env_spec, args_cli):
        captured["env_spec"] = env_spec
        captured["args_cli"] = args_cli
        return expected_arena_env

    monkeypatch.setattr(legacy_graph_environment_cli, "arena_env_from_graph_spec", get_arena_env)

    def get_builder(arena_env, builder_cfg, hydra_overrides, rebuild_index, policy_cfg):
        captured["arena_env"] = arena_env
        captured["builder_cfg"] = builder_cfg
        captured["hydra_overrides"] = hydra_overrides
        captured["rebuild_index"] = rebuild_index
        captured["policy_cfg"] = policy_cfg
        return expected_builder

    # Patched by name so ArenaEnvBuilder is imported when the test runs rather than at
    # collection, which would pull Isaac Lab modules in before SimulationApp starts.
    monkeypatch.setattr("isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder", get_builder)

    run.environment_builder.env_cfg_override = {"episode_length_s": 1.0}
    builder = legacy_graph_environment_cli.build_arena_builder_from_legacy_graph(
        run.environment,
        environment_builder=run.environment_builder,
        hydra_overrides=overrides_from_dict(run.variations),
        rebuild_index=1,
        policy_cfg=run.policy,
    )

    assert builder is expected_builder
    assert captured["arguments"] == legacy_environment_args_to_cli_args(
        {"environment": run.environment.env_spec_path, **run.environment.per_run_overrides}
    )
    assert captured["args_cli"] is parsed_args
    assert captured["env_spec"] == str(graph_path)
    # The Run's typed builder config crosses the boundary directly, so device and
    # language_instruction never round-trip through the argparse namespace.
    assert captured["builder_cfg"] is run.environment_builder
    assert captured["builder_cfg"].device == "cuda:1"
    assert captured["arena_env"] is expected_arena_env
    assert captured["hydra_overrides"] == ["light.intensity.enabled=true"]
    assert captured["rebuild_index"] == 1
    assert captured["policy_cfg"] is run.policy
    assert captured["builder_cfg"].env_cfg_override == {"episode_length_s": 1.0}
