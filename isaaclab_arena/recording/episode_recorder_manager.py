# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import inspect
import json
import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from pathlib import Path
from prettytable import PrettyTable
from typing import Any

from isaaclab.managers import ManagerBase, ManagerTermBase, ManagerTermBaseCfg
from isaaclab.utils.configclass import configclass

CORE_FIELD_NAMES = frozenset({
    "job_name",
    "env_id",
    "episode_in_env",
    "seed",
    "success",
    "episode_length",
    "language_instruction",
    "timestamp",
    "replay_source_record_index",
})
"""Top-level record fields written by the manager. Term names must not use them."""


@configclass
class EpisodeRecorderTermCfg(ManagerTermBaseCfg):
    """Configuration for an episode recorder term."""

    func: Callable[..., dict[str, Any] | None] = MISSING
    """The callable that records this term's output for one finishing episode.

    Invoked as ``func(env, env_id, **params)`` for the env whose episode just finished. It returns a
    JSON object, which the manager stores in the episode's record under the term's name, or ``None``
    to leave the term out of that record. It may also be an ``EpisodeRecorderTerm`` subclass, which the
    manager creates once and steps every control step.
    """


class EpisodeRecorderTerm(ManagerTermBase):
    """Episode recorder term that can follow its episode step by step.

    Subclasses keep one entry of state per env, update it in ``post_step()``, clear it in ``reset()``,
    and return the finished episode's output from ``__call__``. Update state in place: the env may be
    stepped inside ``torch.inference_mode()`` and reset outside it.
    """

    def post_step(self, env) -> None:
        """Update the per-env state after a control step. Does nothing by default."""

    def __call__(self, env, env_id: int, **params) -> dict[str, Any] | None:
        """Return the output for env_id's finished episode, or None to leave it out of the record."""
        raise NotImplementedError


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
        self._class_term_cfgs: list[EpisodeRecorderTermCfg] = []
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

    @property
    def has_post_step_terms(self) -> bool:
        """Whether any term follows its episode step by step."""
        return bool(self._class_term_cfgs)

    def post_step(self) -> None:
        """Let every class-based term update its state after a control step."""
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.post_step(self._env)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
        """Clear the state of every class-based term for the envs that start a new episode."""
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return {}

    def record_pre_reset(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Write one record for each finished episode in env_ids.

        Args:
            env_ids: The env ids being reset (tensor, sequence, or ``None`` for all envs).
        """
        for env_id in self._env_ids_to_list(env_ids):
            record: dict[str, Any] = {"job_name": self._job_name, **self._core_fields(env_id)}
            for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
                output = term_cfg.func(self._env, env_id, **term_cfg.params)
                if output is None:
                    continue
                self._assert_json_object(term_name, output)
                record[term_name] = output
            self._append_record(record)

    def _core_fields(self, env_id: int) -> dict[str, Any]:
        """Return the core fields of env_id's finished episode."""
        env = self._env
        success = None
        if "success" in env.termination_manager.active_terms:
            success = bool(env.termination_manager.get_term("success")[env_id].item())
        fields = {
            "env_id": env_id,
            "episode_in_env": env.get_episode_index(env_id),
            "seed": env.cfg.seed,
            "success": success,
            "episode_length": int(env.episode_length_buf[env_id].item()),
            "language_instruction": env.get_language_instruction(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        scheduler = env.variation_replay_scheduler
        if scheduler is not None:
            fields["replay_source_record_index"] = scheduler.source_record_index_for_env(env_id)
        return fields

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append one record to the output JSONL (one object per line); no-op if no path was set."""
        if self._output_path is None:
            return
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def _env_ids_to_list(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        """Return environment IDs as a list of ints; ``None`` selects all environments."""
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
            assert (
                term_name not in CORE_FIELD_NAMES
            ), f"Episode recorder term name '{term_name}' is a core record field; choose another name."
            # Validate the term's func/params.
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if inspect.isclass(term_cfg.func) or isinstance(term_cfg.func, ManagerTermBase):
                term_class = term_cfg.func if inspect.isclass(term_cfg.func) else type(term_cfg.func)
                assert issubclass(
                    term_class, EpisodeRecorderTerm
                ), f"Episode recorder term '{term_name}' is a class; it must derive from EpisodeRecorderTerm."
                self._class_term_cfgs.append(term_cfg)

    @staticmethod
    def _assert_json_object(term_name: str, output: Any) -> None:
        """Check that ``term_name`` returned a JSON object without NaN or infinity, naming the term if not.

        This points at the offending term rather than surfacing a cryptic error later, when the record
        or the experiment result is written.
        """
        assert isinstance(
            output, dict
        ), f"Episode recorder term '{term_name}' must return a dict or None, got {type(output).__name__}."
        try:
            json.dumps(output, allow_nan=False)
        except TypeError as e:
            raise TypeError(
                f"Episode recorder term '{term_name}' returned non-JSON-serializable output ({output!r}): {e}"
            ) from e
        except ValueError as e:
            raise ValueError(f"Episode recorder term '{term_name}' returned NaN or infinity ({output!r}): {e}") from e
