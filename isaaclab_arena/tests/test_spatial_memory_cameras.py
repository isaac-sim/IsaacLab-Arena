# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera checks of the spatial-memory suite: image observation variants and the analytic visibility model."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

CAMERA_ENV_NAMES = [
    "spatial_memory_mug_past_door",
    "spatial_memory_dishwasher_rack",
    "spatial_memory_machine_tending",
]


def _test_image_observation_variants(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_observation_variants

    for env_name in CAMERA_ENV_NAMES:
        check_observation_variants(env_name, enable_cameras=True)
    return True


def _test_visibility_matches_rendering(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.spatial_memory import check_visibility_matches_rendering

    for env_name in CAMERA_ENV_NAMES:
        check_visibility_matches_rendering(env_name)
    return True


@pytest.mark.with_cameras
def test_spatial_memory_image_observation_variants():
    assert run_function_with_persistent_simulation_app(_test_image_observation_variants, enable_cameras=True)


@pytest.mark.with_cameras
def test_spatial_memory_visibility_matches_rendering():
    assert run_function_with_persistent_simulation_app(_test_visibility_matches_rendering, enable_cameras=True)
