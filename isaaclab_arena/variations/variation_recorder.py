# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase

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
        # Build-time (all-envs) draw; applies to every episode of every env.
        self._build_time_sample: Any = None
        self._has_shared_build_time_sample = False
        # Build-time per-env draws remain fixed across episodes.
        self._build_time_samples_by_env: dict[int, Any] = {}

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

    def record_buildtime_sample(self, sample: Any, env_ids: Sequence[int] | None = None) -> None:
        """Record build-time values shared by all environments or fixed for the supplied ids.

        Args:
            sample: One shared row, or one row per environment id.
            env_ids: Environment ids for per-environment sampling; otherwise None.
        """
        assert self.cfg.sample_per_environment == (
            env_ids is not None
        ), f"Variation '{self.name}' build-time environment ids must agree with sample_per_environment."
        if env_ids is not None:
            assert len(sample) == len(env_ids) and len(env_ids) > 0, (
                f"Variation '{self.name}' build-time draw requires one sample per environment id "
                "and at least one environment."
            )
            assert all(
                type(env_id) is int and env_id >= 0 for env_id in env_ids
            ), f"Variation '{self.name}' build-time environment ids must be non-negative integers."
            assert len(set(env_ids)) == len(
                env_ids
            ), f"Variation '{self.name}' build-time draw contains duplicate environment ids."
            assert (
                not self._has_shared_build_time_sample
            ), f"Variation '{self.name}' already recorded a shared build-time sample."
            assert not self._build_time_samples_by_env.keys() & set(
                env_ids
            ), f"Variation '{self.name}' already recorded a build-time sample for one of these environments."
            for env_id, value in zip(env_ids, sample, strict=True):
                self._build_time_samples_by_env[env_id] = value
            return
        assert (
            len(sample) == 1
        ), f"Variation '{self.name}' build-time draw expected a single sample for all envs; got {len(sample)}."
        assert (
            not self._has_shared_build_time_sample and not self._build_time_samples_by_env
        ), f"Variation '{self.name}' already recorded a build-time sample."
        self._build_time_sample = sample[0]
        self._has_shared_build_time_sample = True

    def sample_for_episode(self, env_id: int, episode_idx: int) -> Any:
        """Return the value drawn for ``env_id``'s ``episode_idx``, or ``None`` if none was drawn.

        Build-time values remain fixed for every episode of their assigned environments.
        """
        key = EnvEpisodeKey(env_id, episode_idx)
        if key in self._samples_by_env_episode:
            return self._samples_by_env_episode[key]
        if env_id in self._build_time_samples_by_env:
            return self._build_time_samples_by_env[env_id]
        return self._build_time_sample


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
                is_build_time = isinstance(variation, BuildTimeVariationBase)
                is_run_time = isinstance(variation, RunTimeVariationBase)
                assert (
                    is_build_time != is_run_time
                ), f"Variation '{variation_key}' must have exactly one build-time or run-time lifecycle."

                # Create a record for the variation
                record = VariationRecord(name=variation_key, cfg=variation.cfg)
                self.records[variation_key] = record

                def on_sample(
                    sample: Any,
                    env_ids: torch.Tensor | None = None,
                    record: VariationRecord = record,
                    is_build_time: bool = is_build_time,
                ) -> None:
                    if isinstance(sample, torch.Tensor):
                        sample = sample.detach().cpu()
                    if is_build_time:
                        env_id_list = env_ids.tolist() if env_ids is not None else None
                        record.record_buildtime_sample(sample, env_id_list)
                    else:
                        assert (
                            record.cfg.sample_per_environment and env_ids is not None
                        ), f"Run-time variation '{record.name}' requires per-environment draws with environment ids."
                        assert self._env is not None, "VariationRecorder needs bind_env() before per-env draws."
                        env_id_list = env_ids.tolist()
                        episode_indices = [self._env.get_episode_index(env_id) for env_id in env_id_list]
                        record.record_runtime_sample(sample, env_id_list, episode_indices)

                variation.add_sample_listener(on_sample)
