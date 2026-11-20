# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_variation_samples(env, env_id: int) -> dict[str, Any] | None:
    """Return the variation values drawn for env_id's finished episode, or None if there are none."""
    recorder = env.variation_recorder
    if recorder is None or not recorder.records:
        return None
    episode_idx = env.get_episode_index(env_id)
    samples: dict[str, Any] = {}
    for key, record in recorder.records.items():
        value = record.sample_for_episode(env_id, episode_idx)
        if value is None:
            continue
        samples[key] = value.tolist() if isinstance(value, torch.Tensor) else value
    return samples or None


def record_placement_sample(env, env_id: int) -> dict[str, Any] | None:
    """Return the relation placement drawn for env_id's finished episode, or None if there is none."""
    recorder = env.variation_recorder
    if recorder is None or recorder.placement_record is None:
        return None
    return recorder.placement_record.sample_for_episode(env_id, env.get_episode_index(env_id))


@configclass
class VariationEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording each variation's per-env sampled value for the episode."""

    func: Callable[..., dict[str, Any] | None] = record_variation_samples


@configclass
class PlacementEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording the relation placement sampled for the episode."""

    func: Callable[..., dict[str, Any] | None] = record_placement_sample
