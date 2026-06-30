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


def world_pose_xyz_quat_xyzw(env, asset_name: str, env_id: int) -> dict[str, list[float]]:
    """World-frame pose of a scene asset for ``env_id`` as ``{pos_w: [x,y,z], quat_w_xyzw: [x,y,z,w]}``.

    This installed Isaac Lab's ``root_quat_w`` is already (x, y, z, w) (per BaseRigidObjectData), so it is
    emitted as-is — no reorder. ``root_pos_w``/``root_quat_w`` are ProxyArray-backed; resolve via
    ``warp.to_torch`` before indexing, as Arena does elsewhere.
    """
    import warp as wp

    asset = env.scene[asset_name]
    pos = wp.to_torch(asset.data.root_pos_w)[env_id]
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[env_id]
    return {"pos_w": [float(v) for v in pos], "quat_w_xyzw": [float(v) for v in quat_xyzw]}


def record_object_poses(env, env_id: int) -> dict[str, Any]:
    """Record world-frame initial (post-reset) and final (episode-end) poses for the configured assets.

    Opt-in: only the asset names in ``env.cfg.pose_snapshot_asset_names`` are captured (empty -> no-op, so
    stock runs are unaffected). ``initial_object_poses`` comes from the start-of-episode snapshot taken in
    ``_reset_idx``; ``final_object_poses`` is read live at episode end. Both are keyed by stable asset name.
    """
    names = list(getattr(env.cfg, "pose_snapshot_asset_names", None) or [])
    if not names:
        return {}
    final = {name: world_pose_xyz_quat_xyzw(env, name, env_id) for name in names}
    initial = env.get_initial_object_pose_snapshot(env_id)
    return {"initial_object_poses": initial, "final_object_poses": final}


def record_gap_provenance(env, env_id: int, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record run/scene provenance (profile, asset channel, resolved URLs, identities, seeds).

    Static scene details come from the environment factory. Runtime and placement seeds come from the
    compiled manager configuration so typed and legacy experiment frontends record the same values.
    """
    if not provenance:
        return {}
    resolved = dict(provenance)
    resolved["placement_seed"] = getattr(env.cfg, "placement_seed", None)
    resolved["seed"] = env.cfg.seed
    return {"gap_provenance": resolved}


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


@configclass
class ObjectPosesEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording initial (post-reset) and final (episode-end) world poses of configured assets."""

    func: Callable[..., dict[str, Any]] = record_object_poses


@configclass
class GapProvenanceEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording run/scene provenance (profile, asset channel, URLs, identities, seeds)."""

    func: Callable[..., dict[str, Any]] = record_gap_provenance
