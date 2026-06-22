# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extensible per-episode results recording via a dedicated manager."""

from __future__ import annotations

import json
import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass
from pathlib import Path
from prettytable import PrettyTable
from typing import Any

from isaaclab.managers import ManagerBase, ManagerTermBaseCfg
from isaaclab.utils import configclass


@dataclass
class EpisodeResultsMetadata:
    """Run-level metadata that cannot be inferred from the env, stamped onto every episode record."""

    job_name: str = "default"
    """Name of the eval job (or run); identifies which job produced the records."""

    language_instruction: str | None = None
    """Job/CLI-level language-instruction override, if any."""


@configclass
class EpisodeRecorderTermCfg(ManagerTermBaseCfg):
    """Configuration for an episode recorder term."""

    func: Callable[..., dict[str, Any]] = MISSING
    """The callable that records this term's fields for one finishing episode.

    Invoked as ``func(env, env_id, **params)`` for the env whose episode just finished, and must
    return a flat, JSON-serializable dict that is merged into the episode's record. It may be a plain
    function or a callable class inheriting from ManagerTermBase (built once by the manager).
    """


class EpisodeRecorderManager(ManagerBase):
    """Buffers one merged per-episode record from its terms; written out as JSONL on request."""

    def __init__(self, cfg: object, env) -> None:
        """Initialize the manager and its episode-recording state.

        Args:
            cfg: The episode recorder manager cfg (or dict of ``EpisodeRecorderTermCfg``).
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[EpisodeRecorderTermCfg] = []
        self._metadata = EpisodeResultsMetadata()
        # One dict per finished episode, in capture order; each merged from all terms' fields.
        self._records: list[dict] = []
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for the episode recorder manager."""
        table = PrettyTable()
        table.title = "Active Episode Recorder Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        return f"<EpisodeRecorderManager> contains {len(self._term_names)} active terms.\n{table.get_string()}\n"

    @property
    def active_terms(self) -> list[str]:
        """Name of active episode recorder terms."""
        return self._term_names

    @property
    def records(self) -> list[dict]:
        """The per-episode records buffered so far, in capture order."""
        return self._records

    def set_metadata(self, metadata: EpisodeResultsMetadata) -> None:
        """Set the run-level metadata stamped onto subsequently recorded episodes."""
        self._metadata = metadata

    def record_pre_reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Buffer one record per finished episode in ``env_ids`` by merging all terms' fields.

        Args:
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        for env_id in self._normalize_env_ids(env_ids):
            # The manager stamps the run-level metadata; terms add the per-episode fields.
            record: dict[str, Any] = {
                "job_name": self._metadata.job_name,
                "language_instruction": self._metadata.language_instruction,
            }
            for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
                fields = term_cfg.func(self._env, env_id, **term_cfg.params)
                collisions = record.keys() & fields.keys()
                assert not collisions, (
                    f"Episode recorder term '{term_name}' redefines fields {collisions} already set"
                    " by the manager or an earlier term."
                )
                record.update(fields)
            self._records.append(record)

    def write(self, output_path: str | Path) -> None:
        """Write all buffered per-episode records to ``output_path`` as JSONL (one object per line)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        """Normalize ``env_ids`` (tensor, sequence, or ``None`` for all envs) to a list of ints."""
        if env_ids is None:
            return list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        return [int(env_id) for env_id in env_ids]

    def _prepare_terms(self) -> None:
        """Build the term callables from the configuration object."""
        for term_name, term_cfg in self.cfg.__dict__.items():
            if term_cfg is None:
                continue
            # Validate the term's func/params and, for callable-class terms, instantiate it once the
            # sim starts (the shared ``ManagerBase`` helper every Isaac Lab manager uses in
            # ``_prepare_terms``, e.g. ``EventManager``). ``min_argc=2`` for the leading env, env_id.
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
