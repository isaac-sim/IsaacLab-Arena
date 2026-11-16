# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spatial-memory task 7: machine tending through an opening."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENV_NAME = "spatial_memory_machine_tending"


def _test_oracle_solves_and_geometry_stays_hidden(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_oracle_solves_with_hidden_geometry

    check_oracle_solves_with_hidden_geometry(
        ENV_NAME, seen_earlier_targets=["opening_left", "opening_right", "opening_top", "clamp_left", "clamp_right"]
    )
    return True


def _test_blind_controller_fails(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_blind_controller_fails

    check_blind_controller_fails(ENV_NAME)
    return True


def _test_deterministic_reset(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_deterministic_reset

    check_deterministic_reset(ENV_NAME)
    return True


def _test_observation_variants(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_observation_variants

    check_observation_variants(ENV_NAME)
    return True


def test_machine_tending_oracle_solves_and_geometry_stays_hidden():
    assert run_function_with_persistent_simulation_app(_test_oracle_solves_and_geometry_stays_hidden)


def test_machine_tending_blind_controller_fails():
    assert run_function_with_persistent_simulation_app(_test_blind_controller_fails)


def test_machine_tending_deterministic_reset():
    assert run_function_with_persistent_simulation_app(_test_deterministic_reset)


def test_machine_tending_observation_variants():
    assert run_function_with_persistent_simulation_app(_test_observation_variants)
