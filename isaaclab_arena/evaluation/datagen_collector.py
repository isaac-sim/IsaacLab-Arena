# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Interface for datagen data collectors, and their CallbackRecorderTerm wiring.

DatagenCollectorBase is implemented by the collecting package (e.g. nvblox_next's
datagen.arena_data_collector.DatagenCollector). build_datagen_callback_handlers adapts
one to the generic CallbackRecorderTermHandlers shape recording.callback_recorder_term
expects, keeping that module free of any datagen-specific knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from isaaclab_arena.evaluation.episode_outcome import EpisodeOutcome, classify_outcome
from isaaclab_arena.recording.callback_recorder_term import CallbackRecorderTermHandlers


class DatagenCollectorBase(ABC):
    """Interface a datagen data collector implements, driven via CallbackRecorderTerm.

    Implementations record per-step data during a policy rollout. on_step fires after
    every env.step (for every env, before any reset that step); on_episode_end fires
    once per env_id right before that env's reset, while its terminal state is still
    intact -- also the place to prepare that env's cameras for the next episode, since
    the reset immediately following flushes any re-aimed poses via IsaacLab's
    num_rerenders_on_reset. finalize/close run at rollout/job teardown.
    """

    @abstractmethod
    def on_step(self, env: Any) -> None:
        """Record one frame for every env.

        Reads env.obs_buf / env.action_manager / env.episode_length_buf directly (no
        args beyond env: state lives on it).
        """

    @abstractmethod
    def on_episode_end(self, env: Any, env_id: int, outcome: EpisodeOutcome = "timeout") -> None:
        """Flush env_id's in-progress episode and prepare its cameras for the next one."""

    @abstractmethod
    def finalize(self, env: Any | None = None) -> None:
        """Flush any in-progress episodes and stop recording. Idempotent."""

    @abstractmethod
    def close(self, env: Any | None = None) -> None:
        """Finalize, then release resources such as spawned cameras. Idempotent."""


def build_datagen_callback_handlers(
    collector: DatagenCollectorBase, env: Any | None = None
) -> CallbackRecorderTermHandlers:
    """Adapt a DatagenCollectorBase to the CallbackRecorderTerm handler shape.

    Args:
        collector: The collector to drive.
        env: If given, on_close calls collector.close(env) with this env instead of
            None (CallbackRecorderTerm's on_close only ever receives a file_path, not
            an env, so callers that need close(env) must bind env here at
            build_handlers time -- see run_execution.py's usage).
    """

    def on_pre_reset(pre_reset_env: Any, env_ids) -> None:
        for env_id in env_ids:
            collector.on_episode_end(pre_reset_env, int(env_id), outcome=classify_outcome(pre_reset_env, int(env_id)))

    return CallbackRecorderTermHandlers(
        on_post_step=collector.on_step,
        on_pre_reset=on_pre_reset,
        on_close=lambda _file_path: collector.close(env),
    )
