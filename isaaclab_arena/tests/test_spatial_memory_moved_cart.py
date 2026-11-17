# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spatial-memory task 10: the cart someone moved."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENV_NAME = "spatial_memory_moved_cart"


def _test_oracle_solves_and_geometry_stays_hidden(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_oracle_solves_with_hidden_geometry

    # The cart moves while hidden by design, so the static-while-hidden check does not apply.
    check_oracle_solves_with_hidden_geometry(ENV_NAME, check_static=False, seen_earlier_targets=["cart"])
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


def test_moved_cart_oracle_solves_and_geometry_stays_hidden():
    assert run_function_with_persistent_simulation_app(_test_oracle_solves_and_geometry_stays_hidden)


def test_moved_cart_blind_controller_fails():
    assert run_function_with_persistent_simulation_app(_test_blind_controller_fails)


def test_moved_cart_deterministic_reset():
    assert run_function_with_persistent_simulation_app(_test_deterministic_reset)


def test_moved_cart_observation_variants():
    assert run_function_with_persistent_simulation_app(_test_observation_variants)
