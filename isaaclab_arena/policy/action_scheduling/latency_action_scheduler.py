# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LatencyActionScheduler: request action chunks ahead of time and deliver them after a fixed latency."""

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from isaaclab_arena.policy.action_scheduling.action_scheduler import ActionScheduler
from isaaclab_arena.utils.timer import Timer

ChunkRequestFn = Callable[[], torch.Tensor]
"""Blocking call that returns the ``(num_requested_envs, horizon, action_dim)`` chunks of a request."""

PrepareRequestsFn = Callable[[list[int]], list[ChunkRequestFn]]
"""Snapshot the given envs' observations and return the calls that fetch their chunks.

The calls run on worker threads and their results are concatenated along the env dimension, so
together they must return one row per requested env, in the order the env ids were given.
"""


@dataclass
class _PendingRequest:
    """Chunks requested on one step for a group of envs."""

    env_ids: list[int]
    futures: list[Future]
    issued_step: int
    rows: torch.Tensor | None = None

    def chunk_rows(self) -> torch.Tensor:
        """Block until every call finished and return the ``(len(env_ids), horizon, action_dim)`` chunks."""
        if self.rows is None:
            with Timer("wait"):
                self.rows = torch.cat([future.result() for future in self.futures], dim=0)
            assert self.rows.shape[0] == len(
                self.env_ids
            ), f"request returned {self.rows.shape[0]} chunks for {len(self.env_ids)} envs"
        return self.rows


class LatencyActionScheduler(ActionScheduler):
    """Chunk scheduler that models a fixed inference latency of ``latency_steps`` control steps.

    A chunk requested with the observation of step ``t`` becomes usable at step ``t + latency_steps``.
    Each env requests its next chunk ``latency_steps`` steps before its current one runs out, so the
    request is in flight while the simulation advances. If the chunk has not arrived when the
    simulation reaches the step it is due, the scheduler blocks; if it arrives earlier, it waits. The
    rollout therefore does not depend on how fast the server answers. Envs without a usable action,
    during the first ``latency_steps`` steps of an episode and whenever a chunk is due later than the
    previous one ends, apply the hold action. With ``latency_steps`` zero requests are made and
    answered on the step they are needed, which is the behavior of ``ActionChunkScheduler``.

    Envs passed to ``park`` request no chunks and apply the hold action until they are reset, so
    envs that finished their last episode do not consume inference while the others complete theirs.
    """

    def __init__(
        self,
        num_envs: int,
        action_chunk_length: int,
        action_horizon: int,
        action_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        *,
        latency_steps: int | None = None,
        max_workers: int = 4,
    ) -> None:
        """Create the scheduler.

        Args:
            num_envs: Number of environments.
            action_chunk_length: Actions executed from each chunk before the next one is needed.
            action_horizon: Actions kept from each returned chunk, at least ``action_chunk_length``.
            action_dim: Size of one action.
            device: Device of the action tensors.
            dtype: Dtype of the action tensors.
            latency_steps: Control steps between a request and its chunk becoming usable. May be
                left None and set later with ``set_latency_steps`` before the first ``get_action``.
            max_workers: Threads that run chunk requests concurrently when ``latency_steps`` is positive.
        """
        assert action_horizon >= action_chunk_length, f"{action_horizon=} must be >= {action_chunk_length=}"
        assert max_workers >= 1, f"max_workers must be at least 1, got {max_workers}"
        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = device
        self._max_workers = max_workers
        self._latency_steps = latency_steps
        self._executor: ThreadPoolExecutor | None = None

        self.current_action_chunk = torch.zeros((num_envs, action_horizon, action_dim), dtype=dtype, device=device)
        # -1 marks an env without a usable chunk.
        self.current_action_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._step = 0
        # Step on which each env issues its next request, or None while a request is in flight.
        self._next_request_step: list[int | None] = [0] * num_envs
        self._pending: list[_PendingRequest | None] = [None] * num_envs
        self._had_chunk_since_reset = [False] * num_envs
        self._parked = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.requests_issued = 0
        """Number of chunk requests issued, one per env and chunk."""
        self.hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        """Steps each env applied the hold action because it had no usable chunk yet."""
        self.stall_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        """Subset of ``hold_steps`` spent after an env's first chunk of the episode ran out."""
        self.parked_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        """Steps each env applied the hold action while parked, not counted in ``hold_steps``."""

    @property
    def latency_steps(self) -> int | None:
        """Control steps between a request and its chunk becoming usable, None until set."""
        return self._latency_steps

    @property
    def has_parked_envs(self) -> bool:
        """Whether any env is parked and therefore needs the hold action on ``get_action``."""
        return bool(self._parked.any())

    def set_latency_steps(self, latency_steps: int) -> None:
        """Set the latency once the control step is known; must happen before the first ``get_action``."""
        assert latency_steps >= 0, f"latency_steps must be non-negative, got {latency_steps}"
        assert self._step == 0, "latency_steps cannot change once actions have been scheduled"
        self._latency_steps = latency_steps

    def get_action(
        self,
        fetch_action_tensor_fn: PrepareRequestsFn,
        hold_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action per env for this step, issuing and collecting chunk requests as they fall due.

        Args:
            fetch_action_tensor_fn: Called with the ids of the envs that request a chunk on this step. It
                snapshots their observations and returns the blocking calls that fetch the chunks; see
                ``PrepareRequestsFn``. Unlike the other schedulers it is not a no-argument fetch.
            hold_action: ``(num_envs, action_dim)`` actions for envs without a usable chunk. Required
                whenever ``latency_steps`` is positive or an env is parked.
        """
        assert self._latency_steps is not None, "set_latency_steps must be called before get_action"
        self._issue_due_requests(fetch_action_tensor_fn)
        self._deliver_due_chunks()
        # With latency_steps >= action_chunk_length the next request is due on the delivered chunk's first step.
        self._issue_due_requests(fetch_action_tensor_fn)

        has_chunk = self.current_action_index >= 0
        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.current_action_chunk[batch_idx, self.current_action_index.clamp(min=0)]
        if not bool(has_chunk.all()):
            assert hold_action is not None, "hold_action is required while some envs wait for their chunk or are parked"
            action = torch.where(has_chunk.unsqueeze(-1), action, hold_action.to(action))
            holding = ~has_chunk & ~self._parked
            self.hold_steps += holding.long()
            stalled = holding & torch.tensor(self._had_chunk_since_reset, device=self.device)
            self.stall_steps += stalled.long()
            self.parked_steps += self._parked.long()

        self.current_action_index[has_chunk] += 1
        exhausted = self.current_action_index >= self.action_chunk_length
        self.current_action_chunk[exhausted] = 0.0
        self.current_action_index[exhausted] = -1
        self._step += 1
        return action

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Drop the given envs' chunks and in-flight requests (all when None); they request anew next step.

        A parked env is unparked, since a reset starts a new episode for it.
        """
        for env_id in self._env_id_list(env_ids):
            self.current_action_chunk[env_id] = 0.0
            self.current_action_index[env_id] = -1
            self._pending[env_id] = None
            self._next_request_step[env_id] = self._step
            self._had_chunk_since_reset[env_id] = False
            self._parked[env_id] = False

    def park(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Stop requesting chunks for the given envs until they are reset.

        Their current chunk and in-flight request are discarded, and ``get_action`` returns the hold
        action for them. Meant for envs that finished their last episode while others are still running.

        Args:
            env_ids: Envs to park; already parked envs are left unchanged.
        """
        for env_id in self._env_id_list(env_ids):
            if self._parked[env_id]:
                continue
            self.current_action_chunk[env_id] = 0.0
            self.current_action_index[env_id] = -1
            self._pending[env_id] = None
            self._next_request_step[env_id] = None
            self._parked[env_id] = True

    def close(self) -> None:
        """Stop the worker threads; requests still in flight are abandoned."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _issue_due_requests(self, prepare_requests_fn: PrepareRequestsFn) -> None:
        """Request chunks for every env whose request step has come, as one group sharing this step's observation."""
        due_env_ids = [
            env_id
            for env_id, request_step in enumerate(self._next_request_step)
            if request_step is not None and request_step <= self._step
        ]
        if not due_env_ids:
            return
        with Timer("dispatch"):
            request_fns = prepare_requests_fn(due_env_ids)
            futures = [self._submit(request_fn) for request_fn in request_fns]
        pending = _PendingRequest(env_ids=due_env_ids, futures=futures, issued_step=self._step)
        for env_id in due_env_ids:
            self._pending[env_id] = pending
            self._next_request_step[env_id] = None
        self.requests_issued += len(due_env_ids)

    def _deliver_due_chunks(self) -> None:
        """Install the chunks whose latency has elapsed for envs that have no chunk to execute."""
        assert self._latency_steps is not None
        for env_id in range(self.num_envs):
            pending = self._pending[env_id]
            if pending is None or self.current_action_index[env_id] >= 0:
                continue
            if pending.issued_step + self._latency_steps > self._step:
                continue
            row = pending.env_ids.index(env_id)
            self.current_action_chunk[env_id] = pending.chunk_rows()[row, : self.action_horizon]
            self.current_action_index[env_id] = 0
            self._pending[env_id] = None
            self._had_chunk_since_reset[env_id] = True
            # Request the next chunk so it is usable on the step after this one's last action.
            self._next_request_step[env_id] = max(
                self._step + self.action_chunk_length - self._latency_steps, self._step
            )

    def _submit(self, request_fn: ChunkRequestFn) -> Future:
        """Run a chunk request on a worker thread, or inline when no latency is modeled."""
        if self._latency_steps == 0:
            future: Future = Future()
            future.set_result(request_fn())
            return future
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="arena-inference")
        return self._executor.submit(request_fn)

    def _env_id_list(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> list[int]:
        """Return the env ids selected by ``env_ids`` as a list; None and slices address the whole batch."""
        all_env_ids = list(range(self.num_envs))
        if env_ids is None:
            return all_env_ids
        if isinstance(env_ids, slice):
            return all_env_ids[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return [int(env_id) for env_id in env_ids.reshape(-1).tolist()]
        return [int(env_id) for env_id in env_ids]
