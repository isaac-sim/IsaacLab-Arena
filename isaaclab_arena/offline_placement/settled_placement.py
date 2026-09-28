# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record solved placements that remain close to their initial poses after physics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.pool_validation import solver_validation_failure, step_placement_physics
from isaaclab_arena.offline_placement.post_physics_validation import (
    PostPhysicsState,
    articulation_link_poses_in_root_frame,
    build_post_physics_validators,
    validate_post_physics,
)
from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
from isaaclab_arena.relations.placement_events import get_placement_pool, get_reset_placement_results
from isaaclab_arena.relations.placement_layouts import PlacementLayouts, validate_replay_reset_policies
from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


@dataclass
class PlacementRecordingResult:
    """Accepted post-physics poses and the outcome of each source candidate."""

    layouts: PlacementLayouts
    """Complete final poses keyed by runtime scene name."""
    accepted_indices: list[tuple[int, int]]
    """Source (environment index, reset batch index) for each output layout, in file order."""
    rejections: dict[tuple[int, int], str]
    """Failure reason for each rejected source (environment index, reset batch index)."""
    validation: list[dict]
    """Solver verdicts and physics-check settings and results, in output layout order."""

    @property
    def attempted(self) -> int:
        """Total source candidates, including solver failures."""
        return len(self.accepted_indices) + len(self.rejections)


def collect_settled_placements(
    env: ManagerBasedEnv,
    num_batches: int,
    params: PlacementRecordingParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingResult:
    """Filter solved layouts with physics and record their final poses.

    Each batch calls env.reset(), consuming one placement per environment.
    The environment remains at its final state on completion or failure.
    Each candidate must pass its required solver checks and every enabled, applicable
    post-physics check. Solver validation is not repeated.

    Args:
        env: Environment with a pooled placement reset event.
        num_batches: Number of resets to sample, independent of pool refills.
        params: Simulation duration, post-physics validators and minimum yield.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.

    Returns:
        Final environment-local poses, source indices and rejected-candidate reasons.
    """
    env = env.unwrapped
    assert num_batches > 0, "num_batches must be positive"
    placement_pool = get_placement_pool(env)
    assert placement_pool is not None, "Recording requires a pooled placement reset event"
    if params is None:
        params = PlacementRecordingParams()
    assert placement_pool.num_envs == env.num_envs, "Placement pool and scene must have the same environment count"
    assets = list(placement_pool.objects)
    for asset in scene_assets or []:
        if asset not in assets:
            assets.append(asset)
    keys = _recording_keys(env, assets)
    embodiment_keys = tuple(asset.get_scene_key() for asset in assets if asset.tags and "embodiment" in asset.tags)
    articulation_keys = [key for key in env.scene.articulations if key not in embodiment_keys]
    validators = build_post_physics_validators(params.validators, articulation_keys)
    accepted: dict[str, list[Pose]] = {key: [] for key in keys}
    accepted_indices: list[tuple[int, int]] = []
    rejections: dict[tuple[int, int], str] = {}
    validation: list[dict] = []
    num_candidates = num_batches * env.num_envs
    for batch_index in range(num_batches):
        layouts, state = _settle_reset(env, keys, articulation_keys, params.num_steps, batch_index, num_batches, render)
        previously_accepted = len(accepted_indices)
        env_ids = state.env_ids
        reports = validate_post_physics(validators, state)
        for env_id, layout in layouts.items():
            failure = solver_validation_failure(layout)
            if failure is not None:
                rejections[env_id, batch_index] = failure
                continue
            failures = [f"{report.check}: {report.reason}" for report in reports[env_id] if report.passed is False]
            if failures:
                rejections[env_id, batch_index] = "; ".join(failures)
                continue
            for key in keys:
                value = state.final_poses[key][env_id].tolist()
                accepted[key].append(Pose(tuple(value[:3]), tuple(value[3:])))
            accepted_indices.append((env_id, batch_index))
            validation.append({
                "pre_physics": dict(layout.validation_results.validation_results),
                "post_physics": [asdict(report) for report in reports[env_id]],
                "sampling": {
                    "num_steps": params.num_steps,
                    "decimation": env.cfg.decimation,
                    "physics_dt_s": env.sim.get_physics_dt(),
                    "embodiment_keys": list(embodiment_keys),
                },
            })
        print(
            f"[recording] batch {batch_index + 1}/{num_batches}: "
            f"{len(layouts)} solutions, {len(env_ids)} passed solver validation, "
            f"{len(accepted_indices) - previously_accepted} passed post-physics validation; "
            f"overall {len(accepted_indices) + len(rejections)}/{num_candidates} validated, "
            f"{len(accepted_indices)} accepted",
            flush=True,
        )
    assert (
        len(accepted_indices) >= params.min_layouts
    ), f"Accepted {len(accepted_indices)} layouts; need {params.min_layouts}. Rejections: {rejections}"
    layouts = PlacementLayouts(accepted)
    layouts.validate_assets(assets)
    return PlacementRecordingResult(layouts, accepted_indices, rejections, validation)


def _settle_reset(
    env: ManagerBasedEnv,
    keys: list[str],
    articulation_keys: list[str],
    num_steps: int,
    batch_index: int,
    num_batches: int,
    render: bool,
) -> tuple[dict[int, PlacementResult], PostPhysicsState]:
    """Reset once and measure the selected layouts before and after physics."""
    env.reset()
    layouts = get_reset_placement_results(env)
    assert set(layouts) == set(range(env.num_envs)), "Reset must place every environment"
    env_ids = [env_id for env_id, layout in layouts.items() if solver_validation_failure(layout) is None]
    initial = {key: env.arena_world.get_pose_e(key) for key in keys}
    initial_links = articulation_link_poses_in_root_frame(env, articulation_keys)
    if env_ids:
        step_placement_physics(env, num_steps, batch_index, num_batches, render, log_progress=True)
    state = PostPhysicsState(
        env=env,
        env_ids=env_ids,
        initial_poses=initial,
        final_poses={key: env.arena_world.get_pose_e(key) for key in keys},
        initial_links=initial_links,
        final_links=articulation_link_poses_in_root_frame(env, articulation_keys),
    )
    return layouts, state


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
    validate_replay_reset_policies([by_key[key] for key in sorted(keys)])
    return sorted(keys)
