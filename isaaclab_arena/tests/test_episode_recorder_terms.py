# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the episode recorder terms in isaaclab_arena.terms.episode_recorders."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _ObjectPositions:
    """Stand-in for ArenaWorld that returns positions set by the test, one row per env."""

    def __init__(self, num_envs: int):
        import torch

        self.position_w = torch.zeros(num_envs, 3)

    def get_position_w(self, scene_key: str):
        assert scene_key == "box"
        return self.position_w.clone()


def _make_displacement_term(num_envs: int = 2):
    from types import SimpleNamespace

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.terms.episode_recorders import ObjectDisplacement

    object_positions = _ObjectPositions(num_envs)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", arena_world=object_positions)
    term_cfg = EpisodeRecorderTermCfg(func=ObjectDisplacement, params={"object_name": "box"})
    return ObjectDisplacement(term_cfg, env), env, object_positions


def _test_displacement_is_the_largest_distance_from_the_first_step(simulation_app):
    import pytest

    term, env, object_positions = _make_displacement_term()
    term.reset()
    # The object spawned at z = 0 and settled at z = 0.9 before the first step; only later motion counts.
    object_positions.position_w[:, 2] = 0.90
    term.post_step()
    # Env 0 is carried 30 cm in 1 mm steps; env 1 is lifted 10 cm and put back.
    for step in range(1, 301):
        object_positions.position_w[0, 0] = 0.001 * step
        object_positions.position_w[1, 2] = 1.00 if 100 <= step < 200 else 0.90
        term.post_step()

    assert term(env, 0, object_name="box") == pytest.approx(0.3, abs=1e-5)
    assert term(env, 1, object_name="box") == pytest.approx(0.1, abs=1e-5)
    return True


def test_displacement_is_the_largest_distance_from_the_first_step():
    assert run_function_with_persistent_simulation_app(_test_displacement_is_the_largest_distance_from_the_first_step)


def _test_reset_starts_a_new_episode_for_the_selected_envs(simulation_app):
    import torch

    term, env, object_positions = _make_displacement_term()
    term.reset()
    term.post_step()
    object_positions.position_w[:, 0] = 0.2
    term.post_step()

    # Env 0 starts a new episode where the object now is; env 1 keeps its episode.
    term.reset(torch.tensor([0]))
    term.post_step()
    object_positions.position_w[:, 0] = 0.21
    term.post_step()

    assert abs(term(env, 0, object_name="box") - 0.01) < 1e-5
    assert abs(term(env, 1, object_name="box") - 0.21) < 1e-5
    return True


def test_reset_starts_a_new_episode_for_the_selected_envs():
    assert run_function_with_persistent_simulation_app(_test_reset_starts_a_new_episode_for_the_selected_envs)
