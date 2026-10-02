# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent variation draws and strict replay of recorded episode values."""

from __future__ import annotations

import hashlib
import json
import torch
from copy import deepcopy
from pathlib import Path
from typing import Any


class VariationReplay:
    """Replay the ``variations`` fields of one Run's episode JSONL, without resampling."""

    def __init__(self, episodes: list[dict[str, Any]], *, build_values: dict[str, Any] | None = None) -> None:
        self._build_values = deepcopy(build_values)
        self._episodes: dict[tuple[int, int], dict[str, Any]] = {}
        for episode in episodes:
            env_id, episode_index = episode.get("env_id"), episode.get("episode_in_env")
            assert type(env_id) is int and env_id >= 0, "Replay requires a nonnegative integer env_id."
            assert (
                type(episode_index) is int and episode_index >= 0
            ), "Replay requires a nonnegative integer episode_in_env."
            key = (env_id, episode_index)
            assert key not in self._episodes, f"Duplicate variation replay episode {key}."
            values = episode.get("variations")
            assert isinstance(values, dict), f"Replay episode {key} requires a variations mapping."
            assert all(isinstance(name, str) and name for name in values), "Replay variation paths must be nonempty."
            self._episodes[key] = deepcopy(values)
        assert self._episodes or self._build_values is not None, "Variation replay requires recorded samples."

    @classmethod
    def from_jsonl(cls, path: str | Path) -> VariationReplay:
        """Read an episode-results or complete sample-trace JSONL from one Run/rebuild."""
        episodes = []
        for line_number, line in enumerate(Path(path).expanduser().read_text().splitlines(), 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            assert isinstance(episode, dict), f"Replay line {line_number} must be an episode object."
            episodes.append(episode)
        if episodes and "schema" in episodes[0]:
            assert episodes[0] == {
                "schema": "arena.variation_samples",
                "version": 1,
            }, "Unsupported variation sample-trace schema."
            return cls._from_sample_rows(episodes[1:])
        return cls(episodes)

    @classmethod
    def _from_sample_rows(cls, rows: list[dict[str, Any]]) -> VariationReplay:
        build_values = {}
        episodes = {}
        runtime_paths = set()
        for row in rows:
            path, scope = row.get("variation"), row.get("scope")
            assert isinstance(path, str) and path, "Sample traces require a variation path."
            assert "value" in row, f"Sample trace for '{path}' has no value."
            if scope == "build":
                assert path not in build_values, f"Duplicate build-time sample for '{path}'."
                build_values[path] = row["value"]
            else:
                assert scope == "runtime", f"Unknown sample scope '{scope}'."
                runtime_paths.add(path)
                env_id, index = row.get("env_id"), row.get("episode_in_env")
                assert type(env_id) is int and env_id >= 0, "Runtime sample requires a nonnegative integer env_id."
                assert (
                    type(index) is int and index >= 0
                ), "Runtime sample requires a nonnegative integer episode_in_env."
                episode = episodes.setdefault(
                    (env_id, index),
                    {
                        "env_id": env_id,
                        "episode_in_env": index,
                        "variations": {},
                    },
                )
                assert path not in episode["variations"], f"Duplicate runtime sample for '{path}' in {(env_id, index)}."
                episode["variations"][path] = row["value"]
        assert runtime_paths.isdisjoint(
            build_values
        ), "A replay variation cannot have both build-time and runtime samples."
        return cls(list(episodes.values()), build_values=build_values)

    def value(self, path: str, key: tuple[int, int] | None) -> Any:
        """Return a recorded runtime value, or the consistent all-episode build-time value."""
        if key is not None:
            assert key in self._episodes, f"Missing variation replay episode {key}."
            values = self._episodes[key]
            assert path in values, f"Missing replay value for '{path}' in episode {key}."
            return deepcopy(values[path])
        if self._build_values is not None:
            assert path in self._build_values, f"Missing build-time replay value for '{path}'."
            return deepcopy(self._build_values[path])
        selected = []
        for episode_key, values in self._episodes.items():
            assert path in values, f"Missing build-time replay value for '{path}' in episode {episode_key}."
            selected.append(values[path])
        encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in selected]
        assert len(set(encoded)) == 1, f"Build-time replay values for '{path}' disagree across episodes."
        return deepcopy(selected[0])


class VariationSamplingContext:
    """Bind variation paths to seeded per-episode draws or recorded values.

    Runtime keys use environment ID and episode index. They are independent of
    reset ordering and unrelated variations, but not reassignment to other env IDs.
    """

    __slots__ = ("seed", "replay", "_env")
    """Keep the live environment binding out of recursive Isaac Lab config validation."""

    def __init__(self, seed: int | None = None, replay: VariationReplay | None = None) -> None:
        assert seed is None or type(seed) is int, "Variation seed must be an integer or None."
        assert seed is not None or replay is not None, "Sampling context requires a seed or replay."
        assert replay is None or isinstance(replay, VariationReplay), "Invalid variation replay source."
        self.seed = seed
        self.replay = replay
        self._env = None

    def __deepcopy__(self, memo: dict[int, Any]) -> VariationSamplingContext:
        """Share the lifecycle binding when Isaac Lab copies event configurations."""
        memo[id(self)] = self
        return self

    def bind_env(self, env) -> None:
        """Bind the environment supplying ``get_episode_index(env_id)`` before runtime draws."""
        self._env = env

    def episode_keys(self, num_samples: int, env_ids) -> list[tuple[int, int] | None]:
        """Resolve sample rows to stable episode keys, with None for one build-time draw."""
        assert type(num_samples) is int and num_samples >= 0, "num_samples must be a nonnegative integer."
        if env_ids is None:
            assert num_samples == 1, "A contextual build-time draw requires exactly one sample."
            return [None]
        ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        assert len(ids) == num_samples, "env_ids must contain one ID per sample."
        assert all(type(env_id) is int and env_id >= 0 for env_id in ids), "env_ids must be nonnegative integers."
        assert len(set(ids)) == len(ids), "A variation draw cannot repeat an environment ID."
        assert self._env is not None, "Bind the sampling context to the environment before runtime draws."
        keys = []
        for env_id in ids:
            index = self._env.get_episode_index(env_id)
            assert type(index) is int and index >= 0, "Episode indices must be nonnegative integers."
            keys.append((env_id, index))
        return keys

    def generator(self, path: str, key: tuple[int, int] | None) -> torch.Generator:
        """Create an independent CPU generator for one variation and episode key."""
        assert self.seed is not None, "Replay-only contexts do not generate random draws."
        identity = [
            "arena-variation-v1",
            self.seed,
            path,
            "build" if key is None else "runtime",
            key,
        ]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**63)
        return torch.Generator(device="cpu").manual_seed(seed)
