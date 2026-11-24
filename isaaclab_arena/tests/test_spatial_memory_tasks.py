# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that every spatial-memory task tests what it claims: solvable, hidden, not solvable by luck, reproducible."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

TASK_ENV_NAMES = [
    "spatial_memory_mug_past_door",
    "spatial_memory_fetch_behind",
    "spatial_memory_box_carry",
    "spatial_memory_reverse_gap",
    "spatial_memory_shelf_bay",
    "spatial_memory_partition_delivery",
    "spatial_memory_dishwasher_rack",
    "spatial_memory_back_of_shelf",
    "spatial_memory_high_shelf",
    "spatial_memory_machine_tending",
    "spatial_memory_kitting",
    "spatial_memory_restock_slot",
    "spatial_memory_moved_cart",
]


def _test_oracle_solves_and_geometry_stays_hidden(simulation_app, env_name: str) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_oracle_solves_with_hidden_geometry

    check_oracle_solves_with_hidden_geometry(env_name)
    return True


def _test_blind_controller_fails(simulation_app, env_name: str) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_blind_controller_fails

    check_blind_controller_fails(env_name)
    return True


def _test_deterministic_reset(simulation_app, env_name: str) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_deterministic_reset

    check_deterministic_reset(env_name)
    return True


def _test_observation_variants(simulation_app, env_name: str) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_observation_variants

    check_observation_variants(env_name)
    return True


@pytest.mark.parametrize("env_name", TASK_ENV_NAMES)
def test_oracle_solves_and_geometry_stays_hidden(env_name: str):
    assert run_function_with_persistent_simulation_app(_test_oracle_solves_and_geometry_stays_hidden, env_name=env_name)


@pytest.mark.parametrize("env_name", TASK_ENV_NAMES)
def test_blind_controller_fails(env_name: str):
    assert run_function_with_persistent_simulation_app(_test_blind_controller_fails, env_name=env_name)


@pytest.mark.parametrize("env_name", TASK_ENV_NAMES)
def test_deterministic_reset(env_name: str):
    assert run_function_with_persistent_simulation_app(_test_deterministic_reset, env_name=env_name)


@pytest.mark.parametrize("env_name", TASK_ENV_NAMES)
def test_observation_variants(env_name: str):
    assert run_function_with_persistent_simulation_app(_test_observation_variants, env_name=env_name)
