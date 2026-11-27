# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the argparse compatibility path that builds graph-spec environments for Runs."""

import sys
from pathlib import Path

from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation import legacy_graph_environment_cli
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants


def test_legacy_graph_builder_keeps_namespace_inside_graph_compatibility(monkeypatch):
    graph_path = Path(TestConstants.test_data_dir) / "pick_and_place_maple_table_env_graph.yaml"
    # The YAML loader builds this from a Run's environment section:
    # {type: <graph path>, enable_cameras: true, object: box, mimic: false}.
    environment_cfg = LegacyGraphEnvironmentCfg(
        env_spec_path=str(graph_path),
        enable_cameras=True,
        per_run_overrides={"enable_cameras": True, "object": "box", "mimic": False},
    )
    environment_builder_cfg = ArenaEnvBuilderCfg()
    policy_cfg = ZeroActionPolicyCfg()
    hydra_overrides = ["light.intensity.enabled=true"]
    expected_arena_env = object()
    expected_builder = object()
    captured = {}
    # The Experiment Runner's command line names the Experiment, not the graph YAML.
    monkeypatch.setattr(sys, "argv", ["experiment_runner.py", "--experiment_cfg", "experiment.yaml"])

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

    builder = legacy_graph_environment_cli.build_arena_builder_from_legacy_graph(
        environment_cfg,
        environment_builder=environment_builder_cfg,
        hydra_overrides=hydra_overrides,
        rebuild_index=1,
        policy_cfg=policy_cfg,
    )

    assert builder is expected_builder
    # The YAML's override flag parses although sys.argv has no --env_spec.
    assert captured["args_cli"].object == "box"
    assert captured["args_cli"].enable_cameras
    assert not captured["args_cli"].mimic
    assert captured["env_spec"] == str(graph_path)
    # The typed builder config crosses the boundary directly instead of round-tripping through the argparse namespace.
    assert captured["builder_cfg"] is environment_builder_cfg
    assert captured["arena_env"] is expected_arena_env
    assert captured["hydra_overrides"] == hydra_overrides
    assert captured["rebuild_index"] == 1
    assert captured["policy_cfg"] is policy_cfg
