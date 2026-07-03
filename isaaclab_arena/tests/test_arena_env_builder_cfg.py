# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed Arena environment-builder configuration boundary."""

import pytest

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg


# TODO(cvolk, 2026-07-03): Delete this compatibility test with arena_env_builder_cfg_from_argparse.
def test_argparse_adapter_maps_builder_configuration():
    """Translate only builder-owned command-line values into the typed config."""
    args_cli = get_isaaclab_arena_cli_parser().parse_args([
        "--num_envs",
        "3",
        "--env_spacing",
        "2.5",
        "--seed",
        "7",
        "--no-solve-relations",
        "--placement_seed",
        "11",
        "--no-resolve_on_reset",
        "--random_yaw_init",
        "--disable_fabric",
        "--mimic",
        "--presets",
        "newton",
        "--device",
        "cpu",
    ])
    args_cli.language_instruction = "pick up the cube"

    cfg = arena_env_builder_cfg_from_argparse(args_cli)

    assert cfg == ArenaEnvBuilderCfg(
        num_envs=3,
        env_spacing=2.5,
        seed=7,
        solve_relations=False,
        placement_seed=11,
        resolve_on_reset=False,
        random_yaw_init=True,
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
