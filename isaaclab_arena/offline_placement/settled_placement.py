# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect solved placements that pass configurable post-physics checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.clutter_preparation import prepare_clutter_settling
from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
from isaaclab_arena.offline_placement.post_physics_validation import (
    build_post_physics_validators,
    default_post_physics_validators,
    evaluate_settled_batch,
)
from isaaclab_arena.offline_placement.settled_batch import sample_and_settle_batch
from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
from isaaclab_arena.relations.placement_events import get_placement_pool
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.offline_placement.post_physics_validation import PlacementOutcome
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass
class SettledPlacementResult:
    """Accepted root poses and outcomes of sampled placements."""

    poses: dict[str, list[Pose]]
    """Runtime root names mapped to accepted environment-local poses in matching list order.

    Positions are xyz in metres; rotations are xyzw quaternions. Each list is empty
    when all sampled candidates are rejected.
    """
    accepted_indices: list[tuple[int, int]]
    """Source (environment index, reset batch index) for each accepted pose list entry."""
    rejections: dict[tuple[int, int], str]
    """Failure reason for each rejected source (environment index, reset batch index)."""
    validation: list[PlacementOutcome]
    """Solver and post-physics results in accepted pose list order."""

    @property
    def attempted(self) -> int:
        """Total source candidates, including solver failures."""
        return len(self.accepted_indices) + len(self.rejections)


def collect_settled_placements(
    env: ManagerBasedEnv,
    num_batches: int,
    params: SettledPlacementParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
    *,
    log_progress: bool = False,
) -> SettledPlacementResult:
    """Sample solved placements, apply physics checks, and return accepted root poses.

    Each batch calls env.reset(), consuming one placement per environment.
    The environment remains at its final state on completion or failure.
    Each candidate must pass its required solver checks and every enabled, applicable
    post-physics check. Solver validation is not repeated. This function does not
    enforce replay restrictions or a minimum accepted count, and does not close env.

    Args:
        env: Environment with a pooled placement reset event.
        num_batches: Number of resets to sample, independent of pool refills.
        params: Simulation duration and post-physics validator settings. If omitted, use
            clutter defaults for ClutterOn scenes and ordinary recording defaults otherwise.
            Explicit settings are used unchanged.
        render: Render the offline physics steps.
        scene_assets: Asset definitions supplementing the pool's embodiment tags.
            Required for ClutterOn: pass the complete get_placement_assets() list for preflight.
            Embodiment tags exclude the asset's scene roots from task-object
            link checks. Other articulations receive those checks; root measurements
            cover all rigid objects and articulations.
        log_progress: Print validator settings, physics-step progress, and per-batch results.

    Returns:
        Final environment-local poses, source indices, typed validation results and
        rejected-candidate reasons. Empty pose lists are returned when no candidate passes.
    """
    env = env.unwrapped
    assert num_batches > 0, "num_batches must be positive"
    placement_pool = get_placement_pool(env)
    assert placement_pool is not None, "Collection requires a pooled placement reset event"
    assert placement_pool.num_envs == env.num_envs, "Placement pool and scene must have the same environment count"
    assets = list(placement_pool.objects)
    for asset in scene_assets or []:
        if asset not in assets:
            assets.append(asset)
    has_clutter = any(get_relation(asset, ClutterOn) is not None for asset in assets)
    if has_clutter:
        assert scene_assets is not None, "Clutter collection requires complete scene_assets from get_placement_assets()"
        prepare_clutter_settling(env, assets)
    if params is None:
        defaults = default_clutter_validators() if has_clutter else default_post_physics_validators()
        params = SettledPlacementParams(validators=defaults)
    keys = sorted(set(env.scene.rigid_objects) | set(env.scene.articulations))
    assert keys, "Collection requires rigid objects or articulations"
    embodiment_keys = set()
    for asset in assets:
        if asset.tags and "embodiment" in asset.tags:
            embodiment_keys.update(asset.get_scene_root_keys())
    articulation_keys = [key for key in env.scene.articulations if key not in embodiment_keys]
    validators = build_post_physics_validators(params.validators, articulation_keys, log_progress=log_progress)
    geometry_keys = set()
    for validator in validators:
        if validator.skip_reason(articulation_keys) is None:
            validator.validate_scene(env, assets)
            geometry_keys.update(validator.get_geometry_keys(assets))
    accepted: dict[str, list[Pose]] = {key: [] for key in keys}
    accepted_indices: list[tuple[int, int]] = []
    rejections: dict[tuple[int, int], str] = {}
    validation: list[PlacementOutcome] = []
    num_candidates = num_batches * env.num_envs
    for batch_index in range(num_batches):
        batch = sample_and_settle_batch(
            env,
            root_keys=keys,
            link_keys=articulation_keys,
            num_env_steps=params.num_steps,
            geometry_keys=sorted(geometry_keys),
            render=render,
            log_progress=log_progress,
        )
        outcomes = evaluate_settled_batch(batch, validators)
        previously_accepted = len(accepted_indices)
        for env_id, outcome in outcomes.items():
            if not outcome.passed:
                rejections[env_id, batch_index] = outcome.rejection_reason
                continue
            for key in keys:
                value = batch.final_root_poses[key][env_id].tolist()
                accepted[key].append(Pose(tuple(value[:3]), tuple(value[3:])))
            accepted_indices.append((env_id, batch_index))
            validation.append(outcome)
        if log_progress:
            print(
                f"[placement] batch {batch_index + 1}/{num_batches}: "
                f"{len(batch.source_layouts)} solutions, {len(batch.env_ids)} passed solver validation, "
                f"{len(accepted_indices) - previously_accepted} passed post-physics validation; "
                f"overall {len(accepted_indices) + len(rejections)}/{num_candidates} validated, "
                f"{len(accepted_indices)} accepted",
                flush=True,
            )
    return SettledPlacementResult(accepted, accepted_indices, rejections, validation)
