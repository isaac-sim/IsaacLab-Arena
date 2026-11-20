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


def _test_withheld_frames_are_blank(simulation_app) -> bool:
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.policies import SpatialMemoryOraclePolicy

    env = make_spatial_memory_env("spatial_memory_moved_cart", num_envs=2, seed=1234, enable_cameras=True)
    try:
        base = env.unwrapped
        episode = get_episode_state(base)
        key = episode.logic.embodiment.camera_obs_key()
        policy = SpatialMemoryOraclePolicy()
        obs, _ = env.reset()
        seen_frame = False
        seen_blank = False
        with torch.inference_mode():
            for _ in range(60):
                obs, _, _, _, _ = env.step(policy.get_action(env, obs))
                image = obs["camera_obs"][key]
                blocked = episode.logic.camera_blocked(base, episode)
                for env_id in range(base.num_envs):
                    is_blank = not bool(image[env_id].any())
                    assert is_blank == bool(blocked[env_id]), "A withheld frame must be blank and only then."
                    seen_blank |= is_blank
                    seen_frame |= not is_blank
        assert seen_frame and seen_blank, "Expected both camera frames (the look) and withheld frames."
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_spatial_memory_withheld_frames_are_blank():
    assert run_function_with_persistent_simulation_app(_test_withheld_frames_are_blank, enable_cameras=True)


@pytest.mark.with_cameras
def test_spatial_memory_image_observation_variants():
    assert run_function_with_persistent_simulation_app(_test_image_observation_variants, enable_cameras=True)


@pytest.mark.with_cameras
def test_spatial_memory_visibility_matches_rendering():
    assert run_function_with_persistent_simulation_app(_test_visibility_matches_rendering, enable_cameras=True)
