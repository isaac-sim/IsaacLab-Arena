# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


@dataclass(frozen=True)
class EnvEpisodeKey:
    """Hashable key identifying one env's draws during one episode."""

    env_id: int
    episode_idx: int


class VariationRecord:
    """Per-variation record of the values drawn for it."""

    def __init__(self, name: str, cfg: VariationBaseCfg) -> None:
        self.name = name
        self.cfg = cfg
        # Run-time draw, one per (env id, episode index).
        self._samples_by_env_episode: dict[EnvEpisodeKey, Any] = {}
        # Build-time (all-envs) draws; these apply to every episode of every env.
        self._build_time_samples: list[Any] = []

    def record_runtime_sample(self, sample: Any, env_ids: Sequence[int], episode_indices: Sequence[int]) -> None:
        """Record each row of ``sample`` against the (env id, episode index) it was drawn for.

        Each (env id, episode index) is expected to be drawn for at most once.

        Args:
            sample: The drawn sample; row ``i`` is the value for the ``i``-th env in ``env_ids``.
            env_ids: The env ids the sample's rows correspond to.
            episode_indices: The episode index each row was drawn during, aligned with ``env_ids``.
        """
        for row, (env_id, episode_idx) in enumerate(zip(env_ids, episode_indices)):
            key = EnvEpisodeKey(env_id, episode_idx)
            assert (
                key not in self._samples_by_env_episode
            ), f"Variation '{self.name}' already recorded a sample for env {env_id}, episode {episode_idx}."
            self._samples_by_env_episode[key] = sample[row]

    def record_buildtime_sample(self, sample: Any) -> None:
        """Record an all-envs (build-time) ``sample`` whose rows apply to every episode of every env."""
        self._build_time_samples.extend(sample[row] for row in range(len(sample)))

    def samples_for_episode(self, env_id: int, episode_idx: int) -> list[Any]:
        """Return every value attributed to ``env_id``'s ``episode_idx`` (empty if none).

        Build-time (all-envs) draws come first, since they apply to every episode, followed by the
        run-time draw made during that episode (if any).
        """
        runtime_sample = self._samples_by_env_episode.get(EnvEpisodeKey(env_id, episode_idx))
        runtime_samples = [] if runtime_sample is None else [runtime_sample]
        return self._build_time_samples + runtime_samples


class VariationRecorder:
    """Records samples drawn by attached variations."""

    def __init__(self) -> None:
        # Records are keyed by: "{asset_name}.{variation_name}"
        self.records: dict[str, VariationRecord] = {}
        # Bound after env construction; supplies the episode index for per-env run-time draws.
        self._env: Any = None

    def bind_env(self, env: Any) -> None:
        """Bind the env so run-time draws can be attributed to its current episode index."""
        self._env = env

    def __getitem__(self, key: str) -> VariationRecord:
        """Return the record stored under "{asset_name}.{variation_name}"."""
        return self.records[key]

    def __contains__(self, key: str) -> bool:
        """Whether a record is stored under "{asset_name}.{variation_name}"."""
        return key in self.records

    def attach(self, variations: dict[str, list[VariationBase]]) -> None:
        """Attach every enabled variation in ``variations`` under "{asset_name}.{variation_name}"."""
        for asset_name, asset_variations in variations.items():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                variation_key = f"{asset_name}.{variation.name}"
                assert (
                    variation_key not in self.records
                ), f"VariationRecorder: asset_name '{variation_key}' is already attached."

                # Create a record for the variation
                record = VariationRecord(name=variation_key, cfg=variation.cfg)
                self.records[variation_key] = record

                def on_sample(
                    sample: Any, env_ids: torch.Tensor | None = None, record: VariationRecord = record
                ) -> None:
                    if isinstance(sample, torch.Tensor):
                        sample = sample.detach().cpu()
                    if env_ids is None:
                        # Build-time / all-envs draw: applies to every episode of every env.
                        record.record_buildtime_sample(sample)
                    else:
                        assert self._env is not None, "VariationRecorder needs bind_env() before per-env draws."
                        env_id_list = env_ids.tolist()
                        episode_indices = [self._env.get_episode_index(env_id) for env_id in env_id_list]
                        record.record_runtime_sample(sample, env_id_list, episode_indices)

                variation.add_sample_listener(on_sample)
