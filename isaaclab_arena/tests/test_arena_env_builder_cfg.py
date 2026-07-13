# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed Arena environment-builder configuration boundary."""

import pytest

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete the argparse adapter tests below with
# arena_env_builder_cfg_from_argparse once runners pass ArenaEnvBuilderCfg directly.
def test_cli_defaults_match_builder_configuration():
    """Keep the manual CLI defaults aligned with the typed builder configuration."""
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])

    assert arena_env_builder_cfg_from_argparse(args_cli) == ArenaEnvBuilderCfg()


def test_argparse_adapter_maps_builder_configuration():
    """Translate only builder-owned command-line values into the typed config."""
    parser = get_isaaclab_arena_cli_parser()
    add_policy_runner_arguments(parser)
    args_cli = parser.parse_args([
        "--policy_type",
        "zero_action",
        "--num_envs",
        "3",
        "--env_spacing",
        "2.5",
        "--seed",
        "7",
        "--no_solve_relations",
        "--placement_seed",
        "11",
        "--no-resolve_on_reset",
        "--disable_fabric",
        "--mimic",
        "--presets",
        "newton",
        "--device",
        "cpu",
        "--language_instruction",
        "pick up the cube",
    ])

    cfg = arena_env_builder_cfg_from_argparse(args_cli)

    assert cfg == ArenaEnvBuilderCfg(
        num_envs=3,
        env_spacing=2.5,
        seed=7,
        solve_relations=False,
        placement_seed=11,
        resolve_on_reset=False,
        disable_fabric=True,
        mimic=True,
        presets="newton",
        device="cpu",
        language_instruction="pick up the cube",
    )


def test_builder_configuration_requires_positive_num_envs():
    """Reject configurations that cannot build any environment instances."""
    with pytest.raises(AssertionError, match="num_envs must be greater than zero"):
        ArenaEnvBuilderCfg(num_envs=0)


def test_builder_copies_custom_placer_params_and_applies_runtime_overrides(monkeypatch):
    """Preserve environment-specific solver tuning while applying run placement controls."""
    from types import SimpleNamespace

    from isaaclab_arena.environments import arena_env_builder as builder_module
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    custom_params = ObjectPlacerParams(
        placement_seed=3,
        resolve_on_reset=True,
        solver_params=RelationSolverParams(clearance_m=0.06),
    )
    captured = {}

    def fake_solve(objects, *, num_envs, placer_params):
        captured.update(objects=objects, num_envs=num_envs, placer_params=placer_params)
        return "placement-event"

    monkeypatch.setattr(builder_module, "solve_and_apply_relation_placement", fake_solve)
    builder = object.__new__(builder_module.ArenaEnvBuilder)
    builder.arena_env = SimpleNamespace(
        scene=SimpleNamespace(get_objects_with_relations=lambda: ["object"]),
        placer_params=custom_params,
    )
    builder.cfg = ArenaEnvBuilderCfg(
        num_envs=4,
        placement_seed=11,
        resolve_on_reset=False,
    )
    builder._placement_event_cfg = None

    builder._solve_relations()

    resolved_params = captured["placer_params"]
    assert captured["objects"] == ["object"]
    assert captured["num_envs"] == 4
    assert resolved_params is not custom_params
    assert resolved_params.solver_params.clearance_m == pytest.approx(0.06)
    assert resolved_params.placement_seed == 11
    assert resolved_params.resolve_on_reset is False
    assert custom_params.placement_seed == 3
    assert custom_params.resolve_on_reset is True
    assert builder._placement_event_cfg == "placement-event"
