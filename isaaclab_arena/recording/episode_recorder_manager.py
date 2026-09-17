# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import torch
from collections.abc import Callable, Iterable, Sequence
from dataclasses import MISSING
from pathlib import Path
from prettytable import PrettyTable
from typing import Any

from isaaclab.managers import ManagerBase, ManagerTermBase, ManagerTermBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class EpisodeRecorderTermCfg(ManagerTermBaseCfg):
    """Configuration for an episode recorder term."""

    func: Callable[..., dict[str, Any]] = MISSING
    """Called as ``func(env, env_id, **params)`` to return JSON-serializable episode fields.

    Top-level keys must not collide; values may nest. Use a function or ManagerTermBase subclass.

    Stateful terms override ``reset(env_ids)`` to capture starting state after reset events.
    On normal resets, capture precedes simulation forward and rendering: read directly written
    state, since derived poses and sensors may still be stale. For ``env.reset_to()``, capture
    follows state restoration, forward and rendering. ``env_ids`` is a tensor or sequence of
    environment IDs, or ``None`` for all environments.
    """


def _record_episode_fields(
    terms: Iterable[tuple[str, EpisodeRecorderTermCfg]], env, env_id: int, record: dict[str, Any]
) -> dict[str, Any]:
    """Merge term fields into record and return it; identify invalid fields by term path."""
    for term_name, term_cfg in terms:
        fields = term_cfg.func(env, env_id, **term_cfg.params)
        collisions = record.keys() & fields.keys()
        assert not collisions, (
            f"Episode recorder term '{term_name}' redefines fields {collisions} already set"
            " by the manager or an earlier term."
        )
        try:
            json.dumps(fields)
        except TypeError as exc:
            raise TypeError(
                f"Episode recorder term '{term_name}' returned non-JSON-serializable fields ({fields!r}): {exc}"
            ) from exc
        record.update(fields)
    return record


def _reset_episode_terms(
    terms: Iterable[tuple[str, EpisodeRecorderTermCfg]], env_ids: Sequence[int] | torch.Tensor | None
) -> None:
    """Reset stateful terms, preserving the failing leaf's path through namespace wrappers."""
    for term_name, term_cfg in terms:
        if isinstance(term_cfg.func, NamespacedEpisodeRecorder):
            # Child failures already include their full path; preserve the original cause.
            term_cfg.func.reset(env_ids=env_ids)
        elif isinstance(term_cfg.func, ManagerTermBase):
            try:
                term_cfg.func.reset(env_ids=env_ids)
            except Exception as exc:
                raise RuntimeError(f"Episode recorder term '{term_name}' failed during reset") from exc


class NamespacedEpisodeRecorder(ManagerTermBase):
    """Group child episode fields under a namespace and forward their reset lifecycle."""

    def __init__(self, cfg: EpisodeRecorderTermCfg, env):
        super().__init__(cfg, env)
        self._term_path: str = cfg.params["namespace"]
        """Registered term path used in diagnostics, such as subtask_0/subtask_1."""
        for name, child_cfg in cfg.params["terms"].items():
            if not isinstance(child_cfg, EpisodeRecorderTermCfg):
                raise TypeError(f"Child episode recorder term '{name}' requires EpisodeRecorderTermCfg")

    def set_term_path(self, term_path: str) -> None:
        """Set this recorder's diagnostic path and propagate it to namespaced children."""
        self._term_path = term_path
        for name, child_cfg in self.cfg.params["terms"].items():
            if isinstance(child_cfg.func, NamespacedEpisodeRecorder):
                child_cfg.func.set_term_path(f"{term_path}/{name}")

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset the selected environments in all stateful child terms."""
        child_terms = ((f"{self._term_path}/{name}", cfg) for name, cfg in self.cfg.params["terms"].items())
        _reset_episode_terms(child_terms, env_ids)

    def __call__(self, env, env_id: int, namespace: str, terms: dict[str, EpisodeRecorderTermCfg]) -> dict[str, Any]:
        """Return one episode's child fields nested under namespace."""
        child_terms = ((f"{self._term_path}/{name}", cfg) for name, cfg in terms.items())
        fields = _record_episode_fields(child_terms, env, env_id, {})
        return {namespace: fields}


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
            _record_episode_fields(zip(self._term_names, self._term_cfgs), self._env, env_id, record)
            self._append_record(record)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict:
        """Reset stateful terms after scene reset; term failures abort the reset with their name.

        Args:
            env_ids: Environments starting new episodes, or None for all. Passed through to terms.

        Returns:
            An empty logging dictionary, following the manager reset contract.
        """
        _reset_episode_terms(zip(self._term_names, self._term_cfgs), env_ids)
        return {}

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
            self._resolve_term_cfg_tree(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_term_cfg_tree(self, term_name: str, term_cfg: EpisodeRecorderTermCfg) -> None:
        """Validate and resolve a term and any episode terms in its parameters."""
        # Parent resolution instantiates child callables; validate their class signatures first.
        if isinstance(term_cfg, EpisodeRecorderTermCfg):
            for key, value in term_cfg.params.items():
                self._resolve_nested_episode_terms(f"{term_name}/{key}", value)
        try:
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
        except TypeError as exc:
            raise TypeError(f"Episode recorder term '{term_name}': {exc}") from exc

    def _resolve_nested_episode_terms(self, path: str, value: Any) -> None:
        """Find episode terms inside the parameter containers supported by Isaac Lab."""
        if isinstance(value, EpisodeRecorderTermCfg):
            self._resolve_term_cfg_tree(path, value)
        elif isinstance(value, dict):
            for key, item in value.items():
                self._resolve_nested_episode_terms(f"{path}/{key}", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._resolve_nested_episode_terms(f"{path}/{index}", item)

    def _process_term_cfg_at_play(self, term_name: str, term_cfg: EpisodeRecorderTermCfg) -> None:
        """Resolve runtime terms and bind namespaced diagnostics to the registered term name."""
        super()._process_term_cfg_at_play(term_name, term_cfg)
        if isinstance(term_cfg.func, NamespacedEpisodeRecorder):
            term_cfg.func.set_term_path(term_name)
