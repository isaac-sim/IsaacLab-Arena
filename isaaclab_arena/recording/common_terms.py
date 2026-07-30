# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import torch
from collections.abc import Callable
from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_core_episode_results(env, env_id: int) -> dict[str, Any]:
    """Record the core per-episode fields for ``env_id``."""
    success = None
    if "success" in env.termination_manager.active_terms:
        success = bool(env.termination_manager.get_term("success")[env_id].item())
    return {
        "env_id": env_id,
        "episode_in_env": env.get_episode_index(env_id),
        "seed": env.cfg.seed,
        "success": success,
        "episode_length": int(env.episode_length_buf[env_id].item()),
        "language_instruction": env.get_language_instruction(),
        "timestamp": datetime.datetime.now().isoformat(),
    }


def record_variation_samples(env, env_id: int) -> dict[str, Any]:
    """Record the variation value drawn for ``env_id``'s finished episode under ``variations``."""
    recorder = env.variation_recorder
    if recorder is None or not recorder.records:
        return {}
    episode_idx = env.get_episode_index(env_id)
    samples: dict[str, Any] = {}
    for key, record in recorder.records.items():
        value = record.sample_for_episode(env_id, episode_idx)
        if value is None:
            continue
        samples[key] = value.tolist() if isinstance(value, torch.Tensor) else value
    return {"variations": samples} if samples else {}


@configclass
class CoreEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording the core per-episode metadata (env id, indices, success, seed, timing)."""

    func: Callable[..., dict[str, Any]] = record_core_episode_results


@configclass
class VariationEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording each variation's per-env sampled value for the episode."""

    func: Callable[..., dict[str, Any]] = record_variation_samples
