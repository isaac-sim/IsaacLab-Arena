# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python per-episode results recorder.

The HDF5 ``RecorderManager`` path stores each term's value as a tensor, so it cannot capture
strings or arbitrary metadata. This recorder buffers one record per episode in memory, captured by
the Arena env's ``_reset_idx`` override at the very top of the reset -- before the auto-reset
overwrites the just-finished episode's success/length state.

The env auto-constructs an empty recorder and only captures into it. The caller (policy_runner /
eval_runner) owns the rest: it sets the run-level ``metadata`` after the fact and requests a write
by passing in an explicit output path.
"""

from __future__ import annotations

import datetime
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResultsMetadata:
    """Run-level metadata that cannot be inferred from the env, stamped onto every episode record."""

    job_name: str = "default"
    """Name of the eval job (or run); identifies which job produced the records."""

    rebuild_idx: int = 0
    """Index of the env rebuild this run belongs to."""

    language_instruction: str | None = None
    """Job/CLI-level language-instruction override, if any."""


class EpisodeResultsRecorder:
    """Buffers core per-episode metadata captured at pre-reset; written out on request."""

    def __init__(self, metadata: EpisodeResultsMetadata | None = None) -> None:
        """Create an empty recorder.

        Args:
            metadata: Run-level metadata stamped onto every record. May be set after construction
                via ``set_metadata``; defaults are used when omitted.
        """
        self._metadata = metadata if metadata is not None else EpisodeResultsMetadata()
        self._records: list[dict] = []
        # The initial reset touches every env before anything has happened; skip it.
        self._first_reset = True
        # Per-env count of completed episodes, keyed by env id.
        self._episode_counts: dict[int, int] = {}
        self.global_episode_index = 0

    def set_metadata(self, metadata: EpisodeResultsMetadata) -> None:
        """Set the run-level metadata stamped onto subsequently recorded episodes."""
        self._metadata = metadata

    @property
    def records(self) -> list[dict]:
        """The episode records buffered so far, in capture order."""
        return self._records

    def record_episode(self, env, env_ids: Any) -> None:
        """Buffer one record per finished episode in ``env_ids``.

        Must be called at the top of ``_reset_idx`` (before reset events / manager resets) so the
        just-finished episode's success flag and episode length are still intact.

        Args:
            env: The Arena env being reset (provides termination manager, cfg, buffers).
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        if env_ids is None:
            env_ids = list(range(env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        else:
            env_ids = list(env_ids)

        # Skip the very first reset (all envs, nothing has happened yet).
        if self._first_reset:
            self._first_reset = False
            return

        for env_id in env_ids:
            env_id = int(env_id)
            success = self._read_success(env, env_id)
            episode_in_env = self._episode_counts.get(env_id, 0)
            self._records.append({
                "job_name": self._metadata.job_name,
                "rebuild_idx": self._metadata.rebuild_idx,
                "global_episode_index": self.global_episode_index,
                "episode_in_env": episode_in_env,
                "env_id": env_id,
                "seed": getattr(env.cfg, "seed", None),
                "success": success,
                "episode_length": int(env.episode_length_buf[env_id].item()),
                "language_instruction": self._metadata.language_instruction,
                "task_description": getattr(env.cfg, "task_description", None),
                "timestamp": datetime.datetime.now().isoformat(),
            })
            self._episode_counts[env_id] = episode_in_env + 1
            self.global_episode_index += 1

    def write(self, output_path: str | Path) -> None:
        """Write all buffered records to ``output_path`` as JSONL (one JSON object per line)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")

    @staticmethod
    def _read_success(env, env_id: int) -> bool | None:
        """Return the episode's success flag, or ``None`` if the task defines no ``success`` term."""
        if "success" in env.termination_manager.active_terms:
            return bool(env.termination_manager.get_term("success")[env_id].item())
        return None
