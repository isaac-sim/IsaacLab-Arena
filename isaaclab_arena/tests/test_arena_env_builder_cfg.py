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


def test_builder_configuration_defaults_to_no_reset_rerenders():
    """Keep the typed builder's reset rendering behavior backward-compatible."""
    assert ArenaEnvBuilderCfg().num_rerenders_on_reset == 0


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
        "--num_rerenders_on_reset",
        "5",
        "--seed",
        "7",
        "--no_solve_relations",
        "--placement_seed",
        "11",
        "--placement_clearance_m",
        "0.0005",
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
        num_rerenders_on_reset=5,
        seed=7,
        solve_relations=False,
        placement_seed=11,
        placement_clearance_m=0.0005,
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


def test_builder_configuration_rejects_negative_reset_rerenders():
    """Reject reset rendering counts that cannot describe a number of steps."""
    with pytest.raises(ValueError, match="num_rerenders_on_reset must be non-negative"):
        ArenaEnvBuilderCfg(num_rerenders_on_reset=-1)


@pytest.mark.parametrize("value", [True, 1.5])
def test_builder_configuration_requires_integral_reset_rerenders(value):
    """Reject booleans and fractional reset rendering counts."""
    with pytest.raises(TypeError, match="num_rerenders_on_reset must be an integer"):
        ArenaEnvBuilderCfg(num_rerenders_on_reset=value)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_builder_configuration_rejects_invalid_placement_clearance(value):
    """Reject relation-solver clearances that cannot describe a physical distance."""
    with pytest.raises(ValueError, match="placement_clearance_m"):
        ArenaEnvBuilderCfg(placement_clearance_m=value)


# ArenaEnvBuilder transitively imports pxr. Run this assertion only after
# run_simulation_app_function has initialized Kit, regardless of pytest's test order.
def _test_compose_manager_cfg_preserves_runtime_builder_values(_simulation_app):
    """Expose effective builder inputs on the runtime configuration."""
    from types import SimpleNamespace

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    scene = SimpleNamespace(
        assets={},
        get_asset_variations=lambda: {},
        get_scene_cfg=lambda: None,
        get_observation_cfg=lambda: None,
        get_events_cfg=lambda: None,
        get_termination_cfg=lambda: None,
        get_rewards_cfg=lambda: None,
        get_curriculum_cfg=lambda: None,
        get_commands_cfg=lambda: None,
    )
    builder = ArenaEnvBuilder(
        IsaacLabArenaEnvironment(name="placement_provenance", scene=scene),
        ArenaEnvBuilderCfg(
            solve_relations=False,
            num_rerenders_on_reset=5,
            placement_seed=71,
            placement_clearance_m=0.0005,
        ),
    )

    env_cfg, _ = builder.compose_manager_cfg()

    assert env_cfg.num_rerenders_on_reset == builder.cfg.num_rerenders_on_reset
    assert env_cfg.placement_seed == builder.cfg.placement_seed
    assert env_cfg.placement_clearance_m == builder.cfg.placement_clearance_m
    return True


def test_compose_manager_cfg_preserves_runtime_builder_values():
    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(
        _test_compose_manager_cfg_preserves_runtime_builder_values,
        headless=True,
    ), "runtime builder value composition test failed"
