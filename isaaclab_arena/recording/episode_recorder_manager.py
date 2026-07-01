# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from pathlib import Path
from prettytable import PrettyTable
from typing import Any

from isaaclab.managers import ManagerBase, ManagerTermBaseCfg
from isaaclab.utils.configclass import configclass


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
    """Records per-episode data, described by terms. Written out as JSONL on request."""

    def __init__(self, cfg: object, env) -> None:
        """Initialize the manager and its episode-recording state.

        Args:
            cfg: The episode recorder manager cfg.
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._term_cfgs: list[EpisodeRecorderTermCfg] = []
        self._job_name: str = "default"
        self._output_path: Path | None = None
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

    def set_job_name(self, job_name: str) -> None:
        """Set the job name stamped onto subsequently recorded episodes."""
        self._job_name = job_name

    def set_output_path(self, output_path: str | Path) -> None:
        """Set the path of the JSONL file that records are appended to as episodes finish.

        Must be called before recording to persist results; without it, finished episodes are not
        written anywhere.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Delete the contents of the file if it already exists by writing an empty string.
        path.write_text("", encoding="utf-8")
        self._output_path = path

    def record_pre_reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Record one record per finished episode.

        This function fires each recording terms' function and merges the results into a single record.

        Args:
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        for env_id in self._normalize_env_ids(env_ids):
            # The manager stamps the job name; terms add the per-episode fields.
            record: dict[str, Any] = {
                "job_name": self._job_name,
            }
            # Fire each recording term's function.
            for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
                fields = term_cfg.func(self._env, env_id, **term_cfg.params)
                collisions = record.keys() & fields.keys()
                assert not collisions, (
                    f"Episode recorder term '{term_name}' redefines fields {collisions} already set"
                    " by the manager or an earlier term."
                )
                self._assert_json_serializable(term_name, fields)
                record.update(fields)
            self._append_record(record)

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append one record to the output JSONL (one object per line); no-op if no path was set."""
        if self._output_path is None:
            return
        with open(self._output_path, "a", encoding="utf-8") as f:
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
            # Validate the term's func/params.
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    @staticmethod
    def _assert_json_serializable(term_name: str, fields: dict[str, Any]) -> None:
        """Check ``term_name``'s recorded ``fields`` are JSON-serializable, failing fast if not.

        This points at the offending term rather than surfacing a cryptic error later at write() time when
        the whole record is serialized.
        """
        try:
            json.dumps(fields)
        except TypeError as e:
            raise TypeError(
                f"Episode recorder term '{term_name}' returned non-JSON-serializable fields ({fields!r}): {e}"
            ) from e
