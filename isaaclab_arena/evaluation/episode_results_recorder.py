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
    """Episode metadata to be stamped onto every episode record."""

    job_name: str = "default"
    """Name of the eval job (or run)."""

    language_instruction: str | None = None
    """Language-instruction override, if any."""


class EpisodeResultsRecorder:
    """Buffers core per-episode metadata captured at pre-reset; written out on request."""

    def __init__(self) -> None:
        self._metadata = EpisodeResultsMetadata()
        self._records: list[dict] = []

    def set_metadata(self, metadata: EpisodeResultsMetadata) -> None:
        """Set the run-level metadata stamped onto subsequently recorded episodes."""
        self._metadata = metadata

    @property
    def records(self) -> list[dict]:
        """The episode records buffered so far, in capture order."""
        return self._records

    def record_episode(self, env, env_ids: Any) -> None:
        """Buffer one record per finished episode in ``env_ids``.

        Must be called by ``_reset_idx`` (before reset events / manager resets) so the just-finished
        episode's success flag and episode length are still intact. The env skips the initial reset
        and advances the episode counters itself; this recorder only reads them.

        Args:
            env: The Arena env being reset (provides termination manager, cfg, buffers, and the
                per-env episode index via ``get_episode_index``).
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        if env_ids is None:
            env_ids = list(range(env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        else:
            env_ids = list(env_ids)

        for env_id in env_ids:
            env_id = int(env_id)
            success = self._read_success(env, env_id)
            episode_in_env = env.get_episode_index(env_id)
            self._records.append({
                "job_name": self._metadata.job_name,
                "episode_in_env": episode_in_env,
                "env_id": env_id,
                "seed": getattr(env.cfg, "seed", None),
                "success": success,
                "episode_length": int(env.episode_length_buf[env_id].item()),
                "language_instruction": self._metadata.language_instruction,
                "timestamp": datetime.datetime.now().isoformat(),
            })

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
