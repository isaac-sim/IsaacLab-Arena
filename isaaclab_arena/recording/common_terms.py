# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import math
import torch
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
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


def _finite_vector(value: Sequence[Real], *, length: int, label: str) -> tuple[float, ...]:
    """Validate and normalize a finite numeric vector."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a {length}-element numeric sequence")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a {length}-element numeric sequence") from exc
    if len(items) != length or any(not isinstance(item, Real) or isinstance(item, bool) for item in items):
        raise ValueError(f"{label} must be a {length}-element numeric sequence")
    normalized = tuple(float(item) for item in items)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"{label} must contain only finite values")
    return normalized


def world_xy_aabb_from_local_bbox(
    pose: Mapping[str, Any],
    local_bbox_min: Sequence[Real],
    local_bbox_max: Sequence[Real],
) -> tuple[list[float], list[float]]:
    """Transform a local 3D AABB by an XYZW world pose and return its world XY AABB."""
    if not isinstance(pose, Mapping):
        raise ValueError("object pose must be a mapping with pos_w and quat_w_xyzw")
    try:
        position = _finite_vector(pose["pos_w"], length=3, label="object pose pos_w")
        quaternion = _finite_vector(pose["quat_w_xyzw"], length=4, label="object pose quat_w_xyzw")
    except KeyError as exc:
        raise ValueError(f"object pose is missing required field {exc.args[0]!r}") from exc

    bbox_min = _finite_vector(local_bbox_min, length=3, label="local bbox min")
    bbox_max = _finite_vector(local_bbox_max, length=3, label="local bbox max")
    if any(minimum >= maximum for minimum, maximum in zip(bbox_min, bbox_max)):
        raise ValueError("local bbox min must be strictly less than local bbox max on every axis")

    qx, qy, qz, qw = quaternion
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("object pose quat_w_xyzw must have non-zero norm")
    qx, qy, qz, qw = (component / norm for component in quaternion)
    rotation = (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)),
        (2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)),
        (2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )
    local_center = tuple((minimum + maximum) * 0.5 for minimum, maximum in zip(bbox_min, bbox_max))
    local_half_extent = tuple((maximum - minimum) * 0.5 for minimum, maximum in zip(bbox_min, bbox_max))
    world_center = tuple(
        position[axis] + sum(rotation[axis][index] * local_center[index] for index in range(3)) for axis in range(3)
    )
    world_half_extent = tuple(
        sum(abs(rotation[axis][index]) * local_half_extent[index] for index in range(3)) for axis in range(3)
    )
    return (
        [world_center[0] - world_half_extent[0], world_center[1] - world_half_extent[1]],
        [world_center[0] + world_half_extent[0], world_center[1] + world_half_extent[1]],
    )


def compute_controlled_gap_observation(
    initial_object_poses: Mapping[str, Any],
    *,
    pick_asset_name: str,
    destination_asset_name: str,
    side: str,
    pick_local_bbox_min: Sequence[Real],
    pick_local_bbox_max: Sequence[Real],
    destination_local_bbox_min: Sequence[Real],
    destination_local_bbox_max: Sequence[Real],
) -> dict[str, Any]:
    """Compute the realized post-reset planar AABB gap for a controlled-gap episode."""
    if not isinstance(initial_object_poses, Mapping):
        raise ValueError("initial_object_poses must be a mapping keyed by asset name")
    if not isinstance(pick_asset_name, str) or not pick_asset_name:
        raise ValueError("pick_asset_name must be a non-empty string")
    if not isinstance(destination_asset_name, str) or not destination_asset_name:
        raise ValueError("destination_asset_name must be a non-empty string")
    if pick_asset_name == destination_asset_name:
        raise ValueError("pick_asset_name and destination_asset_name must be different")
    if side not in {"positive_y", "negative_y"}:
        raise ValueError(f"side must be 'positive_y' or 'negative_y', got {side!r}")

    missing_names = [name for name in (pick_asset_name, destination_asset_name) if name not in initial_object_poses]
    if missing_names:
        raise ValueError(f"initial_object_poses is missing required assets: {missing_names}")

    pick_min, pick_max = world_xy_aabb_from_local_bbox(
        initial_object_poses[pick_asset_name], pick_local_bbox_min, pick_local_bbox_max
    )
    destination_min, destination_max = world_xy_aabb_from_local_bbox(
        initial_object_poses[destination_asset_name], destination_local_bbox_min, destination_local_bbox_max
    )
    axis_gap = [
        max(pick_min[axis] - destination_max[axis], destination_min[axis] - pick_max[axis], 0.0) for axis in range(2)
    ]
    signed_side_gap = pick_min[1] - destination_max[1] if side == "positive_y" else destination_min[1] - pick_max[1]
    return {
        "pick_asset_name": pick_asset_name,
        "destination_asset_name": destination_asset_name,
        "side": side,
        "axis_gap_m": axis_gap,
        "planar_aabb_gap_m": math.hypot(*axis_gap),
        "signed_side_gap_m": signed_side_gap,
    }


def record_controlled_gap_observation(
    env,
    env_id: int,
    *,
    pick_asset_name: str,
    destination_asset_name: str,
    side: str,
    pick_local_bbox_min: Sequence[Real],
    pick_local_bbox_max: Sequence[Real],
    destination_local_bbox_min: Sequence[Real],
    destination_local_bbox_max: Sequence[Real],
) -> dict[str, Any]:
    """Record the realized controlled gap from the episode's actual post-reset pose snapshot."""
    snapshot_getter = getattr(env, "get_initial_object_pose_snapshot", None)
    if not callable(snapshot_getter):
        raise ValueError("controlled-gap recorder requires get_initial_object_pose_snapshot(env_id)")
    observation = compute_controlled_gap_observation(
        snapshot_getter(env_id),
        pick_asset_name=pick_asset_name,
        destination_asset_name=destination_asset_name,
        side=side,
        pick_local_bbox_min=pick_local_bbox_min,
        pick_local_bbox_max=pick_local_bbox_max,
        destination_local_bbox_min=destination_local_bbox_min,
        destination_local_bbox_max=destination_local_bbox_max,
    )
    return {"controlled_gap_observation": observation}


def record_gap_provenance(env, env_id: int, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record run/scene provenance (profile, asset channel, resolved URLs, identities, seeds) verbatim.

    ``provenance`` is a build-time dict supplied as a term param; emitted under ``gap_provenance`` so staging
    artifacts cannot be mistaken for production. Independent of CAP's scalar target_specs encoding.
    """
    return {"gap_provenance": provenance} if provenance else {}


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
class ControlledGapObservationEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording the realized post-reset planar gap for an opt-in controlled-gap episode.

    Configure ``params`` with the pick/destination asset names, side, and each asset's local bbox min/max.
    """

    func: Callable[..., dict[str, Any]] = record_controlled_gap_observation


@configclass
class GapProvenanceEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording run/scene provenance (profile, asset channel, URLs, identities, seeds)."""

    func: Callable[..., dict[str, Any]] = record_gap_provenance
