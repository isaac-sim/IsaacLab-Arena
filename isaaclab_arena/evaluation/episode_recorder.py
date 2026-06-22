# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extensible per-episode results recording via a dedicated manager.

The HDF5 ``RecorderManager`` path stores each term's value as a tensor, so it cannot capture
strings or arbitrary metadata. This module adds a separate, pure-Python ``EpisodeRecorderManager``
that runs its terms at pre-reset (driven by the env's ``_reset_idx`` override, before the auto-reset
overwrites the just-finished episode's success/length state), merges each term's fields into one
record per episode, and writes them out as JSONL on request. The base ``RecorderManager`` keeps
handling trajectories/metrics (tensors) unchanged.

Terms are declared in an ``EpisodeRecorderManagerCfg`` (Isaac Lab manager-term style): each
``EpisodeRecorderTermCfg`` carries a callable -- a function ``func(env, context, **params) -> dict``
or a ``ManagerTermBase`` callable class -- that the manager builds and invokes per finished episode.
The core metadata Arena records out of the box is just one such callable
(``record_core_episode_results``). Run-level metadata and the per-episode indices are stamped by the
manager itself, not by the terms.
"""

from __future__ import annotations

import datetime
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


@dataclass
class EpisodeContext:
    """Identity of a finishing episode, tracked by the env and passed to every term callable."""

    env_id: int
    """The env id whose episode just finished."""

    episode_in_env: int
    """Zero-based index of this episode within its env (the env's count of completed episodes)."""


@configclass
class EpisodeRecorderTermCfg(ManagerTermBaseCfg):
    """Configuration for an episode recorder term."""

    func: Callable[..., dict[str, Any]] = MISSING
    """The callable that records this term's fields for one finishing episode.

    Invoked as ``func(env, context, **params)`` where ``context`` is an EpisodeContext, and must
    return a flat, JSON-serializable dict that is merged into the episode's record. It may be a plain
    function or a callable class inheriting from ManagerTermBase (built once by the manager).
    """


def record_core_episode_results(env, context: EpisodeContext) -> dict[str, Any]:
    """Record the core env-derived per-episode fields (seed, success, length, timestamp).

    This is the default term Arena records and the reference example for custom term callables.
    Run-level metadata and episode indices are added by the manager, so they are absent here.
    """
    success = None
    if "success" in env.termination_manager.active_terms:
        success = bool(env.termination_manager.get_term("success")[context.env_id].item())
    return {
        "seed": getattr(env.cfg, "seed", None),
        "success": success,
        "episode_length": int(env.episode_length_buf[context.env_id].item()),
        "timestamp": datetime.datetime.now().isoformat(),
    }


@configclass
class EpisodeRecorderManagerCfg:
    """Configuration for the ``EpisodeRecorderManager``: one ``EpisodeRecorderTermCfg`` per term.

    Subclass and add fields to record more per-episode metadata; set a field to ``None`` to disable
    a term inherited from a base config.
    """

    core: EpisodeRecorderTermCfg = EpisodeRecorderTermCfg(func=record_core_episode_results)
    """The default term recording the core per-episode metadata."""


class EpisodeRecorderManager(ManagerBase):
    """Buffers one merged per-episode record from its terms, captured at pre-reset; written on request.

    The manager builds a term callable per ``EpisodeRecorderTermCfg`` in its config and, for each
    finished episode, stamps the run-level metadata and the env id / episode index then merges every
    term's fields into a single record. The per-env episode index is owned by the env (read via
    ``get_episode_index``), which also skips the initial reset and advances the counters.
    ``record_pre_reset`` must be called by the env's ``_reset_idx`` (before reset events / manager
    resets) so the just-finished episode's success flag and episode length are still intact.
    """

    def __init__(self, cfg: object, env) -> None:
        """Initialize the manager and its episode-recording state.

        Args:
            cfg: The ``EpisodeRecorderManagerCfg`` (or dict of ``EpisodeRecorderTermCfg``).
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[EpisodeRecorderTermCfg] = []
        self._metadata = EpisodeResultsMetadata()
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

        Must be called by the env's ``_reset_idx`` (before reset events / manager resets). The env
        skips the initial reset and owns the per-env episode index (read via ``get_episode_index``).

        Args:
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        for env_id in self._normalize_env_ids(env_ids):
            context = EpisodeContext(env_id=env_id, episode_in_env=self._env.get_episode_index(env_id))
            # The manager stamps run-level metadata and the env id / episode index; terms add the rest.
            record: dict[str, Any] = {
                "job_name": self._metadata.job_name,
                "episode_in_env": context.episode_in_env,
                "env_id": context.env_id,
                "language_instruction": self._metadata.language_instruction,
            }
            for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
                fields = term_cfg.func(self._env, context, **term_cfg.params)
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
        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, EpisodeRecorderTermCfg):
                raise TypeError(
                    f"Configuration for the episode recorder term '{term_name}' is not of type"
                    f" EpisodeRecorderTermCfg. Received: '{type(term_cfg)}'."
                )
            # Resolve/validate func and (deferred to sim play) instantiate callable-class terms.
            # min_argc=2 since term callables take the env and the EpisodeContext before any params.
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
