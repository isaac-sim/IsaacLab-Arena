# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete episode recorder term callables and the default manager configuration.

These are the ready-made terms Arena records out of the box. ``record_core_episode_results`` is the
reference example for writing custom term callables, and ``EpisodeRecorderManagerCfg`` wires the
default terms in. The framework pieces (manager, term-cfg base) live in ``episode_recorder_manager``.
"""

from __future__ import annotations

import datetime
import torch
from typing import Any

from isaaclab.utils import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_core_episode_results(env, env_id: int) -> dict[str, Any]:
    """Record the core env-derived per-episode fields (seed, success, length, timestamp).

    This is the default term Arena records and the reference example for custom term callables.
    Run-level metadata and episode indices are added by the manager, so they are absent here.
    """
    success = None
    if "success" in env.termination_manager.active_terms:
        success = bool(env.termination_manager.get_term("success")[env_id].item())
    return {
        "seed": getattr(env.cfg, "seed", None),
        "success": success,
        "episode_length": int(env.episode_length_buf[env_id].item()),
        "timestamp": datetime.datetime.now().isoformat(),
    }


def record_variation_samples(env, env_id: int) -> dict[str, Any]:
    """Record the variation sample value active for ``env_id``'s finished episode, per variation.

    Reads the per-env latest value tracked by the env's variation recorder (the value sampled at the
    start of the just-finished episode). Returns ``{}`` when the env has no variation recorder or no
    enabled variations.
    """
    # Read the private attribute directly: the public ``variation_recorder`` property logs a warning
    # when unset, which would be noisy since this term runs for every finished episode.
    recorder = getattr(env, "_variation_recorder", None)
    if recorder is None or not recorder.records:
        return {}
    samples = {key: _to_jsonable(record.value_for_env(env_id)) for key, record in recorder.records.items()}
    return {"variations": samples}


def _to_jsonable(value: Any) -> Any:
    """Convert a recorded variation value to a JSON-serializable form (tensors -> nested lists)."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


@configclass
class EpisodeRecorderManagerCfg:
    """Configuration for the ``EpisodeRecorderManager``: one ``EpisodeRecorderTermCfg`` per term.

    Subclass and add fields to record more per-episode metadata; set a field to ``None`` to disable
    a term inherited from a base config.
    """

    core: EpisodeRecorderTermCfg = EpisodeRecorderTermCfg(func=record_core_episode_results)
    """The default term recording the core per-episode metadata."""

    variations: EpisodeRecorderTermCfg = EpisodeRecorderTermCfg(func=record_variation_samples)
    """The default term recording each variation's per-env sampled value for the episode."""
