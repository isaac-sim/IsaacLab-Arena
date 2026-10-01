# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared asset checks and JSONL output for placement recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.offline_placement.post_physics_validation import PlacementOutcome
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.pose import Pose


@dataclass
class PlacementRecordingSummary:
    """Recording output and candidate acceptance counts."""

    output: Path | None
    """Written JSONL path, or None when too few candidates were accepted."""
    accepted: int
    """Number of candidates that passed all required checks."""
    attempted: int
    """Total sampled candidates, including solver failures."""
    rejections: dict[tuple[int, int], str]
    """Rejection reasons keyed by source (environment index, reset batch index)."""


def _merge_validator_configs(base: dict[str, dict], overrides: dict[str, dict]) -> dict[str, dict]:
    """Combine default validator targets with partial Hydra overrides."""
    merged = {name: dict(configuration) for name, configuration in base.items()}
    for name, override in overrides.items():
        if name in merged:
            combined = dict(merged[name])
            combined.update(override)
            merged[name] = combined
        else:
            merged[name] = dict(override)
    return merged


def resolve_settle_params(
    assets: list[PlaceableAsset], params: PlacementRecordingParams | None
) -> PlacementRecordingParams:
    """Return settle params, merging clutter validators when the scene uses ClutterOn.

    Args:
        assets: Placement and scene assets checked for clutter relations.
        params: User settle settings, or None for scene-appropriate defaults.
    """
    from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
    from isaaclab_arena.offline_placement.post_physics_validation import default_post_physics_validators
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.relations.relations import ClutterOn, get_relation

    has_clutter = any(get_relation(asset, ClutterOn) is not None for asset in assets)
    if params is None:
        validators = default_clutter_validators() if has_clutter else default_post_physics_validators()
        return PlacementRecordingParams(validators=validators)
    if not has_clutter:
        return params
    merged = _merge_validator_configs(default_clutter_validators(), params.validators)
    return replace(params, validators=merged)


def collect_layouts_until_count(
    env: ManagerBasedEnv,
    target_count: int,
    max_batches: int,
    params: PlacementRecordingParams | None = None,
    *,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> tuple[dict[str, list[Pose]], list[PlacementOutcome], int, dict[tuple[int, int], str]]:
    """Sample reset batches until the target accept count or batch budget is reached.

    Args:
        env: Built environment with a pooled placement reset event.
        target_count: Number of accepted layouts to collect.
        max_batches: Maximum outer reset-and-settle rounds.
        params: Physics duration and post-physics validators.
        render: Render offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.

    Returns:
        Accepted poses, validation outcomes, attempt count and rejection reasons.
    """
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements

    assert target_count > 0 and max_batches > 0, "Target count and batch budget must be positive"
    poses: dict[str, list[Pose]] = {}
    outcomes: list[PlacementOutcome] = []
    rejections: dict[tuple[int, int], str] = {}
    attempted = 0
    for batch_index in range(max_batches):
        result = collect_settled_placements(env, 1, params, render=render, scene_assets=scene_assets, log_progress=True)
        attempted += result.attempted
        for (env_id, _), reason in result.rejections.items():
            rejections[env_id, batch_index] = reason
        remaining = target_count - len(outcomes)
        for key, values in result.poses.items():
            poses.setdefault(key, []).extend(values[:remaining])
        outcomes.extend(result.validation[:remaining])
        print(
            f"[recording] batch {batch_index + 1}/{max_batches}: {len(outcomes)}/{target_count} collected",
            flush=True,
        )
        if len(outcomes) >= target_count:
            break
    return poses, outcomes, attempted, rejections


def validate_recording_assets(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Require scene root ownership and reset policies compatible with reusable recordings.

    Args:
        env: Built environment whose writable physics roots will be recorded.
        assets: Placement and unplaced scene assets represented in the recording.
    """
    from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners
    from isaaclab_arena.relations.placement_layouts import validate_root_reset_for_cached_layouts
    from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

    env = env.unwrapped
    assert not has_heterogeneous_objects(assets), "Resolve object sets before recording reusable layouts"
    keys = set(env.scene.rigid_objects) | set(env.scene.articulations)
    owners = get_scene_root_owners(assets)
    assert keys <= owners.keys(), f"Pass scene_assets for unplaced scene roots: {keys - owners.keys()}"
    recorded_assets = []
    for asset in assets:
        owned_keys = set(asset.get_scene_root_keys())
        selected = owned_keys.intersection(keys)
        if selected or (not asset.is_anchor and asset.get_spatial_relations()):
            assert (
                owned_keys and owned_keys <= keys
            ), f"'{asset.name}' needs writable physics roots: {owned_keys - keys}"
        if selected:
            assert (
                get_relation(asset, RandomAroundSolution) is None
            ), f"'{asset.name}': remove RandomAroundSolution for cached replay"
            recorded_assets.append(asset)
    assert keys, "Recording requires rigid objects or articulations"
    validate_root_reset_for_cached_layouts(recorded_assets)


def write_settled_layouts(
    env: ManagerBasedEnv,
    output: str | Path,
    assets: list[PlaceableAsset],
    poses: dict[str, list[Pose]],
    outcomes: list[PlacementOutcome],
    num_steps: int,
) -> None:
    """Write accepted root poses and their validation reports as episode JSONL.

    The caller validates recording compatibility before sampling and owns env.
    This function does not reset, step or close it. Existing output is never overwritten.
    """
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts

    layouts = PlacementLayouts(poses)
    layouts.validate_assets(assets)
    embodiment_keys = []
    for asset in assets:
        if asset.tags and "embodiment" in asset.tags:
            embodiment_keys.extend(asset.get_scene_root_keys())
    sampling = {
        "num_steps": num_steps,
        "decimation": env.unwrapped.cfg.decimation,
        "physics_dt_s": env.unwrapped.sim.get_physics_dt(),
        "embodiment_keys": embodiment_keys,
    }
    validation = []
    for outcome in outcomes:
        validation.append({
            "pre_physics": outcome.pre_physics,
            "post_physics": [asdict(report) for report in outcome.post_physics],
            "sampling": sampling,
        })
    layouts.write_episode_jsonl(output, source="settled", validation=validation)
