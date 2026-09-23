# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record solved placements that remain close to their initial poses after physics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.pool_validation import iter_pool_validation
from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
from isaaclab_arena.offline_placement.scene_snapshot import SceneSnapshot, articulation_link_poses_in_root_frame
from isaaclab_arena.offline_placement.validators import (
    PostPhysicsState,
    build_post_physics_validators,
    validate_post_physics,
)
from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_layouts import PlacementLayouts
from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


@dataclass
class PlacementRecordingResult:
    """Accepted post-physics poses and the outcome of each source candidate."""

    layouts: PlacementLayouts
    """Complete final poses keyed by runtime scene name."""
    accepted_indices: list[tuple[int, int]]
    """Source (environment index, queue index) for each output layout, in file order."""
    rejections: dict[tuple[int, int], str]
    """Failure reason for each rejected source (environment index, queue index)."""
    validation: list[dict]
    """Solver verdicts and physics-check settings and results, in output layout order."""

    @property
    def attempted(self) -> int:
        """Total source candidates, including solver failures."""
        return len(self.accepted_indices) + len(self.rejections)


def collect_settled_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer,
    params: PlacementRecordingParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingResult:
    """Filter solved layouts with physics and record their final poses.

    Preserve the source pool and restore scene roots, joints and actuator targets.
    Each candidate must pass its required solver checks and every enabled, applicable
    post-physics check. Solver validation is not repeated.

    Args:
        env: Initialized environment containing the pool's scene assets.
        placement_pool: Solved layouts grouped by absolute environment index.
        params: Simulation duration, post-physics validators and minimum yield.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.

    Returns:
        Final environment-local poses, source indices and rejected-candidate reasons.
    """
    env = env.unwrapped
    if params is None:
        params = PlacementRecordingParams()
    assert placement_pool.num_envs == env.num_envs, "Placement pool and scene must have the same environment count"
    assets = list(placement_pool.objects)
    for asset in scene_assets or []:
        if asset not in assets:
            assets.append(asset)
    keys = _recording_keys(env, assets)
    validators = build_post_physics_validators(params.validators, env)
    snapshot = SceneSnapshot(env)
    initial_links = articulation_link_poses_in_root_frame(env)
    accepted: dict[str, list[Pose]] = {key: [] for key in keys}
    accepted_indices: list[tuple[int, int]] = []
    rejections: dict[tuple[int, int], str] = {}
    validation: list[dict] = []
    batches = iter_pool_validation(
        env,
        placement_pool,
        keys,
        settle_params=PhysicsSettleParams(num_steps=params.num_steps),
        snapshot=snapshot,
        skip_failed=True,
        render=render,
    )
    try:
        for batch in batches:
            env_ids = [env_id for env_id in batch.layouts if env_id not in batch.skipped_layouts]
            state = PostPhysicsState(
                env=env,
                env_ids=env_ids,
                initial_poses=batch.initial_poses,
                final_poses=batch.final_poses,
                initial_links=initial_links,
                final_links=articulation_link_poses_in_root_frame(env),
            )
            reports = validate_post_physics(validators, state)
            for env_id, layout in batch.layouts.items():
                if env_id in batch.skipped_layouts:
                    rejections[env_id, batch.index] = batch.skipped_layouts[env_id]
                    continue
                failures = [f"{report.check}: {report.reason}" for report in reports[env_id] if report.passed is False]
                if failures:
                    rejections[env_id, batch.index] = "; ".join(failures)
                    continue
                for key in keys:
                    value = batch.final_poses[key][env_id].tolist()
                    accepted[key].append(Pose(tuple(value[:3]), tuple(value[3:])))
                accepted_indices.append((env_id, batch.index))
                validation.append({
                    "pre_physics": dict(layout.validation_results.validation_results),
                    "post_physics": [asdict(report) for report in reports[env_id]],
                    "sampling": {
                        "num_steps": params.num_steps,
                        "decimation": env.cfg.decimation,
                        "physics_dt_s": env.sim.get_physics_dt(),
                    },
                })
        assert (
            len(accepted_indices) >= params.min_layouts
        ), f"Accepted {len(accepted_indices)} layouts; need {params.min_layouts}. Rejections: {rejections}"
        layouts = PlacementLayouts(accepted)
        layouts.validate_assets(assets)
        return PlacementRecordingResult(layouts, accepted_indices, rejections, validation)
    finally:
        snapshot.restore(env)


def _recording_keys(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> list[str]:
    """Return writable scene roots with concrete asset definitions for replay."""
    assert not has_heterogeneous_objects(assets), "Resolve object sets before recording reusable layouts"
    keys = set(env.scene.rigid_objects) | set(env.scene.articulations)
    by_key = {asset.get_scene_key(): asset for asset in assets}
    assert len(by_key) == len(assets), "Recording assets must have distinct scene keys"
    assert keys <= by_key.keys(), f"Pass scene_assets for unplaced scene roots: {keys - by_key.keys()}"
    for asset in assets:
        key = asset.get_scene_key()
        assert (
            asset.is_anchor or not asset.get_spatial_relations() or key in keys
        ), f"'{key}' needs a writable physics root"
        if key in keys:
            assert (
                get_relation(asset, RandomAroundSolution) is None
            ), f"'{key}': remove RandomAroundSolution for cached replay"
    assert keys, "Recording requires rigid objects or articulations"
    return sorted(keys)
