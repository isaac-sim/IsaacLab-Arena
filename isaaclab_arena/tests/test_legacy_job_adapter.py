# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

from isaaclab_arena.evaluation import legacy_job_adapter
from isaaclab_arena.evaluation.job_manager import Job
from isaaclab_arena.evaluation.legacy_job_adapter import adapt_legacy_eval_config
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicyCfg
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg


def test_legacy_jobs_become_concrete_experiment_configs():
    legacy_config = {
        "jobs": [{
            "name": "maple_table",
            "arena_env_args": {
                "environment": "pick_and_place_maple_table",
                "embodiment": "droid_rel_joint_pos",
                "pick_up_object": "mustard_bottle_hot3d_robolab",
                "num_envs": 4,
                "env_spacing": 2.5,
                "enable_cameras": True,
            },
            "policy_type": "zero_action",
            "policy_config_dict": {},
            "num_steps": 20,
            "language_instruction": "Move the bottle.",
            "variations": {"light": {"intensity": {"enabled": True}}},
        }]
    }

    (experiment,) = adapt_legacy_eval_config(legacy_config, device="cuda:1")

    assert experiment.name == "maple_table"
    assert experiment.environment == PickAndPlaceMapleTableEnvironmentCfg(
        enable_cameras=True,
        embodiment="droid_rel_joint_pos",
        pick_up_object="mustard_bottle_hot3d_robolab",
    )
    assert experiment.environment_builder.num_envs == 4
    assert experiment.environment_builder.env_spacing == 2.5
    assert experiment.environment_builder.device == "cuda:1"
    assert experiment.environment_builder.language_instruction == "Move the bottle."
    assert experiment.policy == ZeroActionPolicyCfg()
    assert experiment.rollout.num_steps == 20
    assert experiment.variations == {"light": {"intensity": {"enabled": True}}}


def test_legacy_graph_environment_stays_in_the_existing_cli_path():
    graph_path = Path(TestConstants.test_data_dir) / "pick_and_place_maple_table_env_graph.yaml"
    legacy_config = {
        "jobs": [{
            "name": "graph_environment",
            "arena_env_args": {
                "environment": str(graph_path),
                "enable_cameras": True,
                "object": "dex_cube",
            },
            "policy_type": "zero_action",
            "num_steps": 2,
        }]
    }

    (experiment,) = adapt_legacy_eval_config(legacy_config, device="cpu")

    assert isinstance(experiment.environment, legacy_job_adapter._LegacyCliEnvironmentCfg)
    assert experiment.environment.arguments == Job.convert_args_dict_to_cli_args_list(
        legacy_config["jobs"][0]["arena_env_args"]
    )


def test_legacy_graph_builder_keeps_namespace_inside_adapter(monkeypatch):
    graph_path = Path(TestConstants.test_data_dir) / "pick_and_place_maple_table_env_graph.yaml"
    (experiment,) = adapt_legacy_eval_config(
        {
            "jobs": [{
                "name": "graph_environment",
                "arena_env_args": {"environment": str(graph_path), "num_envs": 2},
                "policy_type": "zero_action",
                "num_steps": 2,
                "variations": {"light": {"intensity": {"enabled": True}}},
            }]
        },
        device="cuda:1",
    )
    parsed_args = SimpleNamespace()
    expected_builder = object()
    captured = {}

    class _Parser:
        def parse_args(self, arguments):
            captured["arguments"] = arguments
            return parsed_args

    monkeypatch.setattr(legacy_job_adapter, "get_isaaclab_arena_environments_cli_parser", lambda: _Parser())

    def get_builder(args_cli, hydra_overrides):
        captured["args_cli"] = args_cli
        captured["hydra_overrides"] = hydra_overrides
        return expected_builder

    monkeypatch.setattr(legacy_job_adapter, "get_arena_builder_from_cli", get_builder)

    builder = legacy_job_adapter._build_arena_builder_from_legacy_cfg(experiment)

    assert builder is expected_builder
    assert captured["arguments"] == experiment.environment.arguments
    assert parsed_args.device == "cuda:1"
    assert parsed_args.language_instruction is None
    assert captured["hydra_overrides"] == ["light.intensity.enabled=true"]
