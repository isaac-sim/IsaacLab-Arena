# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ActionChunkScheduler and SyncedBatchActionScheduler.

Pure torch — no Isaac Sim, no remote server. Verifies the per-env stepping,
reset semantics, and (for the synced variant) the hold-on-wait behavior.
"""

from __future__ import annotations

import torch

NUM_ENVS = 2
ACTION_CHUNK_LENGTH = 4
ACTION_HORIZON = 4
ACTION_DIM = 3


def _make_chunk() -> torch.Tensor:
    """Deterministic (NUM_ENVS, ACTION_HORIZON, ACTION_DIM) tensor with unique values per slot."""
    return torch.arange(NUM_ENVS * ACTION_HORIZON * ACTION_DIM, dtype=torch.float).reshape(
        NUM_ENVS, ACTION_HORIZON, ACTION_DIM
    )


# ----------------------------- ActionChunkScheduler ------------------------------


def test_action_chunk_scheduler_steps_through_chunk_and_refetches_when_exhausted():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # First call: every env needs a chunk → exactly one fetch, action is slot 0.
    a0 = scheduler.get_action(fetch)
    assert fetch_calls == 1
    assert a0.shape == (NUM_ENVS, ACTION_DIM)
    torch.testing.assert_close(a0, chunk[:, 0])

    # Subsequent calls within the chunk step without re-fetching.
    for k in range(1, ACTION_CHUNK_LENGTH):
        action = scheduler.get_action(fetch)
        torch.testing.assert_close(action, chunk[:, k])
    assert fetch_calls == 1

    # After draining the chunk, every env should be flagged as needing a new one.
    assert scheduler.env_requires_new_chunk.all()

    # Next call triggers a new fetch.
    scheduler.get_action(fetch)
    assert fetch_calls == 2


def test_action_chunk_scheduler_reset_marks_all_envs_for_refetch():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Drive one step so the buffer is populated and indices have advanced.
    scheduler.get_action(fetch)
    assert fetch_calls == 1
    assert not scheduler.env_requires_new_chunk.any()

    scheduler.reset()

    assert scheduler.env_requires_new_chunk.all()
    assert (scheduler.current_action_index == -1).all()
    assert (scheduler.current_action_chunk == 0.0).all()

    # Next get_action must trigger a refetch.
    scheduler.get_action(fetch)
    assert fetch_calls == 2


def test_action_chunk_scheduler_reset_per_env_only_touches_selected_envs():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Populate both envs through a normal fetch+step.
    scheduler.get_action(fetch)
    assert fetch_calls == 1

    scheduler.reset(torch.tensor([1]))

    # env 0 keeps its (advanced) state; env 1 is flagged for refetch.
    assert scheduler.env_requires_new_chunk.tolist() == [False, True]
    assert scheduler.current_action_index.tolist() == [1, -1]

    # Next call refetches because env 1 needs a chunk; env 0 must not be overwritten.
    env0_buffer_before = scheduler.current_action_chunk[0].clone()
    scheduler.get_action(fetch)
    assert fetch_calls == 2
    torch.testing.assert_close(scheduler.current_action_chunk[0], env0_buffer_before)


# --------------------------- SyncedBatchActionScheduler --------------------------


def test_synced_batch_scheduler_fetches_only_when_all_envs_need_a_chunk():
    from isaaclab_arena.policy.action_scheduling import SyncedBatchActionScheduler

    scheduler = SyncedBatchActionScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # First call: every env needs a chunk → exactly one fetch.
    a0 = scheduler.get_action(fetch, hold)
    assert fetch_calls == 1
    torch.testing.assert_close(a0, chunk[:, 0])

    # Subsequent calls within the chunk: no further fetches, just stepping.
    for k in range(1, ACTION_CHUNK_LENGTH):
        a = scheduler.get_action(fetch, hold)
        torch.testing.assert_close(a, chunk[:, k])
    assert fetch_calls == 1

    # Chunk exhausted → next call triggers a new fetch.
    scheduler.get_action(fetch, hold)
    assert fetch_calls == 2


def test_synced_batch_scheduler_holds_waiting_envs_after_partial_reset():
    from isaaclab_arena.policy.action_scheduling import SyncedBatchActionScheduler

    scheduler = SyncedBatchActionScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()
    hold = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Bring everyone in sync, take one step.
    scheduler.get_action(fetch, hold)  # fetch happens here
    assert fetch_calls == 1

    # Reset env 1; env 0 still has chunk to play.
    scheduler.reset(torch.tensor([1]))

    a = scheduler.get_action(fetch, hold)
    # No new fetch — env 0 is not yet exhausted, so .all() is False.
    assert fetch_calls == 1
    # env 0 advances to chunk[0, 1]; env 1 is waiting and gets the hold action.
    torch.testing.assert_close(a[0], chunk[0, 1])
    torch.testing.assert_close(a[1], hold[1])


# --------------------------- LatencyActionScheduler ----------------------------


class _ChunkServer:
    """Fake chunk server that hands out one chunk per requested env and records the request schedule."""

    def __init__(self, chunk: torch.Tensor | None = None):
        self.chunk = _make_chunk() if chunk is None else chunk
        self.requests: list[list[int]] = []
        self.step = 0

    def prepare_requests(self, env_ids: list[int]):
        self.requests.append((self.step, list(env_ids)))
        rows = self.chunk[env_ids]
        return [lambda rows=rows: rows]

    def steps(self, scheduler, hold, count: int) -> list[torch.Tensor]:
        actions = []
        for _ in range(count):
            actions.append(scheduler.get_action(self.prepare_requests, hold))
            self.step += 1
        return actions


def _make_latency_scheduler(latency_steps: int, action_chunk_length: int = ACTION_CHUNK_LENGTH):
    from isaaclab_arena.policy.action_scheduling import LatencyActionScheduler

    return LatencyActionScheduler(
        NUM_ENVS, action_chunk_length, ACTION_HORIZON, ACTION_DIM, device="cpu", latency_steps=latency_steps
    )


def test_latency_scheduler_with_zero_latency_matches_the_chunk_scheduler():
    """At zero latency every action and every request step equals ActionChunkScheduler's."""
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    chunk = _make_chunk()
    reference = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    reference_fetch_steps = []
    scheduler = _make_latency_scheduler(latency_steps=0)
    server = _ChunkServer(chunk)

    for step in range(2 * ACTION_CHUNK_LENGTH + 1):
        if step == ACTION_CHUNK_LENGTH + 1:
            reference.reset(torch.tensor([1]))
            scheduler.reset(torch.tensor([1]))

        def reference_fetch() -> torch.Tensor:
            reference_fetch_steps.append(step)
            return chunk

        expected = reference.get_action(reference_fetch)
        (actual,) = server.steps(scheduler, None, 1)
        torch.testing.assert_close(actual, expected)

    assert [step for step, _ in server.requests] == reference_fetch_steps
    assert scheduler.hold_steps.tolist() == [0, 0]


def test_latency_scheduler_holds_then_requests_ahead_of_the_chunk_end():
    """With d < H the env holds d steps, then requests each next chunk d steps before the current one ends."""
    latency_steps = 2
    scheduler = _make_latency_scheduler(latency_steps)
    server = _ChunkServer()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    actions = server.steps(scheduler, hold, 3 * ACTION_CHUNK_LENGTH)

    for action in actions[:latency_steps]:
        torch.testing.assert_close(action, hold)
    for k, action in enumerate(actions[latency_steps:]):
        torch.testing.assert_close(action, server.chunk[:, k % ACTION_CHUNK_LENGTH])
    # Requests on step 0, then d steps before each chunk ends: 2 + 4 - 2 = 4, 8.
    assert [step for step, _ in server.requests] == [0, 4, 8]
    assert scheduler.hold_steps.tolist() == [latency_steps] * NUM_ENVS
    assert scheduler.stall_steps.tolist() == [0] * NUM_ENVS


def test_latency_scheduler_stalls_between_chunks_when_latency_exceeds_the_chunk():
    """With d > H the next request goes out on the chunk's first step and the env stalls d - H steps."""
    latency_steps = ACTION_CHUNK_LENGTH + 1
    scheduler = _make_latency_scheduler(latency_steps)
    server = _ChunkServer()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    actions = server.steps(scheduler, hold, latency_steps + 2 * ACTION_CHUNK_LENGTH + 1)

    first_chunk = range(latency_steps, latency_steps + ACTION_CHUNK_LENGTH)
    stall = range(first_chunk.stop, first_chunk.stop + latency_steps - ACTION_CHUNK_LENGTH)
    second_chunk = range(stall.stop, stall.stop + ACTION_CHUNK_LENGTH)
    for step, action in enumerate(actions):
        if step in first_chunk:
            torch.testing.assert_close(action, server.chunk[:, step - first_chunk.start])
        elif step in second_chunk:
            torch.testing.assert_close(action, server.chunk[:, step - second_chunk.start])
        else:
            torch.testing.assert_close(action, hold)
    assert [step for step, _ in server.requests][:3] == [0, first_chunk.start, second_chunk.start]
    assert scheduler.stall_steps.tolist() == [latency_steps - ACTION_CHUNK_LENGTH] * NUM_ENVS


def test_latency_scheduler_with_latency_equal_to_the_chunk_never_stalls():
    """With d == H the next chunk is requested on the first step of the current one and arrives just in time."""
    scheduler = _make_latency_scheduler(ACTION_CHUNK_LENGTH)
    server = _ChunkServer()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    actions = server.steps(scheduler, hold, 3 * ACTION_CHUNK_LENGTH)

    for k, action in enumerate(actions[ACTION_CHUNK_LENGTH:]):
        torch.testing.assert_close(action, server.chunk[:, k % ACTION_CHUNK_LENGTH])
    assert scheduler.stall_steps.tolist() == [0] * NUM_ENVS


def test_latency_scheduler_reset_discards_the_in_flight_request_of_that_env_only():
    """A reset env ignores the chunk it requested, requests again next step and holds; the other env continues."""
    latency_steps = 2
    scheduler = _make_latency_scheduler(latency_steps)
    server = _ChunkServer()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    server.steps(scheduler, hold, 1)  # step 0: both envs request
    scheduler.reset(torch.tensor([1]))
    actions = server.steps(scheduler, hold, 3)  # steps 1-3

    # Env 0 receives its chunk on step 2; env 1 re-requested on step 1 and receives on step 3.
    torch.testing.assert_close(actions[1][0], server.chunk[0, 0])
    torch.testing.assert_close(actions[1][1], hold[1])
    torch.testing.assert_close(actions[2][0], server.chunk[0, 1])
    torch.testing.assert_close(actions[2][1], server.chunk[1, 0])
    assert server.requests == [(0, [0, 1]), (1, [1])]
    assert scheduler.hold_steps.tolist() == [2, 3]


def test_latency_scheduler_concatenates_several_request_calls_in_env_order():
    """A request may be split into several calls; their rows are assigned to the envs in the order given."""
    scheduler = _make_latency_scheduler(latency_steps=1)
    chunk = _make_chunk()
    hold = torch.zeros(NUM_ENVS, ACTION_DIM)

    def prepare_requests(env_ids: list[int]):
        return [lambda env_id=env_id: chunk[env_id : env_id + 1] for env_id in env_ids]

    scheduler.get_action(prepare_requests, hold)
    action = scheduler.get_action(prepare_requests, hold)

    torch.testing.assert_close(action, chunk[:, 0])
    scheduler.close()
