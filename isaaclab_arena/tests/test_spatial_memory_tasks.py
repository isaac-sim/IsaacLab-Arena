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
    "spatial_memory_dogleg_exit",
    "spatial_memory_turning_pocket",
    "spatial_memory_two_view_exit",
    "spatial_memory_funnel_retreat",
    "spatial_memory_staggered_lips",
    "spatial_memory_curved_channel",
    "spatial_memory_post_turn",
    "spatial_memory_column_grid",
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


def _test_two_gates_require_separate_views(simulation_app):
    import math
    import torch

    from isaaclab_arena_environments.spatial_memory.g1_adapter import G1Adapter
    from isaaclab_arena_environments.spatial_memory.tasks.two_view_exit import TwoViewExitLogic

    logic = TwoViewExitLogic(None, G1Adapter())
    camera = logic.embodiment.camera_model()
    half_diagonal = math.hypot(camera.width_px / (2 * camera.focal_x_px), camera.height_px / (2 * camera.focal_y_px))
    # A cone enclosing the frustum extends from the pinhole to a disc at the far clip plane.
    diameter = camera.far_m * max(2 * half_diagonal, math.hypot(1, half_diagonal))
    poses = logic.sample_layout(torch.Generator().manual_seed(11)).part_poses_e
    names = logic.part_names()
    near = names.index("outer_left")
    far = names.index("inner_left")
    sizes = logic.part_specs()
    gap = abs(float(poses[near, 0] - poses[far, 0])) - (sizes[near].size_xyz[0] + sizes[far].size_xyz[0]) / 2
    assert gap > diameter, "A single camera frustum could contain evidence from both gate rows."
    return True


def test_two_gates_require_separate_views():
    assert run_function_with_persistent_simulation_app(_test_two_gates_require_separate_views)
