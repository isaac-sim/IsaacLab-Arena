# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test episode assignments and limits without a simulation instance."""

import pytest

from isaaclab_arena.environments.episode_scheduler import EpisodeScheduler


def test_episode_scheduler_waits_for_every_started_episode() -> None:
    episode_scheduler = EpisodeScheduler(3, episode_limit=5)
    assert episode_scheduler.start_episodes([2, 0, 1]).tolist() == [0, 1, 2]
    assert [episode_scheduler.get_global_episode_index(env_id) for env_id in range(3)] == [0, 1, 2]

    episode_scheduler.finish_episodes([1, 2])
    assert episode_scheduler.num_episodes_completed == 2
    assert episode_scheduler.start_episodes([1, 2]).tolist() == [1, 2]
    assert [episode_scheduler.get_global_episode_index(env_id) for env_id in range(3)] == [0, 3, 4]
    assert episode_scheduler.num_episodes_started == 5
    assert not episode_scheduler.is_complete

    episode_scheduler.finish_episodes([1])
    assert episode_scheduler.start_episodes([1]).numel() == 0
    assert episode_scheduler.get_global_episode_index(1) is None
    assert episode_scheduler.active_episode_mask.tolist() == [True, False, True]
    assert episode_scheduler.num_episodes_completed == 3
    assert not episode_scheduler.is_complete

    episode_scheduler.finish_episodes([0, 2])
    assert episode_scheduler.is_complete
    assert episode_scheduler.num_episodes_completed == 5
    assert episode_scheduler.start_episodes([0, 1, 2]).numel() == 0
    assert [episode_scheduler.get_episode_index_in_env(env_id) for env_id in range(3)] == [0, 1, 1]


@pytest.mark.parametrize("episode_limit", [0, 1, 2])
def test_episode_scheduler_starts_only_requested_episodes(episode_limit: int) -> None:
    episode_scheduler = EpisodeScheduler(3, episode_limit=episode_limit)
    started_episode_env_ids = episode_scheduler.start_episodes([0, 1, 2])
    assert started_episode_env_ids.tolist() == list(range(episode_limit))
    episode_scheduler.finish_episodes(started_episode_env_ids)
    assert episode_scheduler.is_complete
    assert episode_scheduler.num_episodes_started == episode_scheduler.num_episodes_completed == episode_limit


def test_episode_scheduler_does_not_start_a_replacement_after_final_episode() -> None:
    episode_scheduler = EpisodeScheduler(1, episode_limit=1)
    episode_scheduler.start_episodes([0])
    episode_scheduler.finish_episodes([0])
    assert episode_scheduler.start_episodes([0]).numel() == 0
    assert episode_scheduler.is_complete
    assert episode_scheduler.get_episode_index_in_env(0) == 0


def test_episode_scheduler_supports_unlimited_episodes() -> None:
    episode_scheduler = EpisodeScheduler(1)
    for episode_index in range(4):
        assert episode_scheduler.start_episodes([0]).tolist() == [0]
        assert episode_scheduler.get_global_episode_index(0) == episode_index
        assert episode_scheduler.get_episode_index_in_env(0) == episode_index
        episode_scheduler.finish_episodes([0])
    assert episode_scheduler.num_episodes_completed == 4
    assert not episode_scheduler.is_complete


def test_episode_scheduler_rejects_invalid_transitions_without_changing_assignments() -> None:
    episode_scheduler = EpisodeScheduler(3)
    episode_scheduler.set_episode_limit(2)
    episode_scheduler.start_episodes([0])
    with pytest.raises(AssertionError, match="active environment"):
        episode_scheduler.start_episodes([0, 1])
    with pytest.raises(AssertionError, match="inactive environment"):
        episode_scheduler.finish_episodes([0, 1])
    with pytest.raises(AssertionError, match="after episodes have started"):
        episode_scheduler.set_episode_limit(3)
    assert episode_scheduler.get_global_episode_index(0) == 0
    assert episode_scheduler.num_episodes_completed == 0
    assert episode_scheduler.num_episodes_started == 1


def test_episode_scheduler_active_mask_cannot_change_assignments() -> None:
    episode_scheduler = EpisodeScheduler(2)
    episode_scheduler.start_episodes([0])
    episode_scheduler.active_episode_mask.fill_(False)
    assert episode_scheduler.active_episode_mask.tolist() == [True, False]
    assert episode_scheduler.get_global_episode_index(0) == 0
