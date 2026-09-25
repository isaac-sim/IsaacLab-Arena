# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_events import (
    get_base_rotation_per_asset,
    get_movable_asset_names,
    get_placement_pool,
    write_layout_to_sim,
)
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils import physics_settle

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.offline_placement.scene_snapshot import SceneSnapshot
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


@dataclass
class PoolValidationBatch:
    """Physics results for one queue index across N simulation environments."""

    index: int
    """Queue index shared by the candidates in this batch."""
    layouts: dict[int, PlacementResult]
    """Source candidates by environment ID, including any skipped solver failures."""
    skipped_layouts: dict[int, str]
    """Reason each unapplied candidate was rejected, by environment ID."""
    initial_poses: dict[str, torch.Tensor]
    """Environment-local pre-physics poses (N, 7), ordered xyz/xyzw; empty without pose capture."""
    final_poses: dict[str, torch.Tensor]
    """Environment-local post-physics poses (N, 7), ordered xyz/xyzw; empty without pose capture."""


def iter_pool_validation(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer,
    object_names: list[str] | None = None,
    *,
    settle_params: PhysicsSettleParams,
    snapshot: SceneSnapshot | None = None,
    skip_failed: bool = False,
    render: bool = False,
) -> Iterator[PoolValidationBatch]:
    """Apply and simulate each pool batch without consuming its queues.

    The scene remains at the yielded poses until iteration resumes. This iterator
    does not modify source checklists or restore state when it closes.

    Args:
        env: Initialized simulation environment.
        placement_pool: Candidate queues indexed by absolute environment ID.
        object_names: Scene roots to capture; None skips pose capture.
        settle_params: Simulation duration; this iterator does not evaluate velocity thresholds.
        snapshot: Optional state restored before each batch; the caller owns final restoration.
        skip_failed: Leave candidates with missing or failed required solver checks unapplied.
        render: Render each physics step.

    Returns:
        Batches containing source candidates, measured initial and final poses.
    """
    env = env.unwrapped
    object_names = object_names or []
    assets = placement_pool.objects
    anchors = set(get_anchor_objects(assets))
    rotations = get_base_rotation_per_asset(assets)
    queues = placement_pool.layouts_per_env()[: env.num_envs]
    for index in range(max((len(queue) for queue in queues), default=0)):
        if snapshot is not None:
            snapshot.restore(env)
        layouts = {}
        skipped_layouts = {}
        env_ids = []
        for env_id, queue in enumerate(queues):
            if index >= len(queue):
                continue
            layout = queue[index]
            layouts[env_id] = layout
            if skip_failed:
                failure = _solver_validation_failure(layout)
                if failure is not None:
                    skipped_layouts[env_id] = failure
                    continue
            write_layout_to_sim(env, env_id, layout, anchors, rotations)
            env_ids.append(env_id)
        if not env_ids:
            yield PoolValidationBatch(index, layouts, skipped_layouts, {}, {})
            continue
        env.scene.write_data_to_sim()
        env.sim.forward()
        initial = {key: env.arena_world.get_pose_e(key) for key in object_names}
        physics_settle.step_physics(env, settle_params.num_steps * env.cfg.decimation, render=render)
        final = {key: env.arena_world.get_pose_e(key) for key in object_names}
        yield PoolValidationBatch(index, layouts, skipped_layouts, initial, final)


def _solver_validation_failure(layout: PlacementResult) -> str | None:
    """Return a rejection reason for incomplete or failed required solver checks."""
    checklist = layout.validation_results
    missing = (checklist.required_checks or set()) - checklist.validation_results.keys()
    if missing:
        return f"missing required solver checks: {', '.join(sorted(missing))}"
    if not layout.success:
        return "solver validation failed"
    return None


def validate_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer | None = None,
    settle_params: PhysicsSettleParams | None = None,
    render: bool = False,
) -> list[tuple[int, int, PlacementValidationResults]] | None:
    """Physics-validate every layout in a placement pool, recording the result on its validation results.

    Steps physics on every stored layout and stamps the ``PHYSICS_SETTLED`` outcome onto that
    layout's ``PlacementValidationResults``.

    Args:
        env: The Isaac Lab env.
        placement_pool: PooledObjectPlacer whose stored layouts are validated. When ``None`` it is derived from
            the env's registered pooled layouts.
        settle_params: Settle-check tuning params. Defaults to
            ``PhysicsSettleParams()`` when omitted.
        render: When True, render each settle step so the sweep is visible in the GUI. Defaults to False.

    Returns:
        ``(env_id, episode_index, checklist)`` for every layout, in ``(env_id, episode_index)`` order,
        or ``None`` when ``placement_pool`` is omitted and the env has no pooled layouts.
    """
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None
    if settle_params is None:
        settle_params = PhysicsSettleParams()

    assets = placement_pool.objects
    anchors = set(get_anchor_objects(assets))
    object_names = get_movable_asset_names(assets, anchors)
    batches = iter_pool_validation(
        env,
        placement_pool,
        settle_params=settle_params,
        render=render,
    )
    results: list[tuple[int, int, PlacementValidationResults]] = []
    for batch in batches:
        env_ids = list(batch.layouts)
        settled = physics_settle.are_all_objects_settled_per_env(
            env, env_ids, object_names, settle_params.lin_vel_thresh, settle_params.ang_vel_thresh
        )
        settled_per_env = dict(zip(env_ids, settled, strict=True))
        for env_id, layout in batch.layouts.items():
            checklist = layout.validation_results
            if PlacementCheck.PHYSICS_SETTLED not in checklist.validation_results:
                checklist.add_validation_check(PlacementCheck.PHYSICS_SETTLED, settled_per_env[env_id])
            results.append((env_id, batch.index, checklist))
    results.sort(key=lambda item: (item[0], item[1]))
    return results


def print_validation_results(results: list[tuple[int, int, PlacementValidationResults]]) -> None:
    """Print each layout's validation results and a pass/fail summary for a pool validation run."""
    if not results:
        print("Placement pool has no layouts to validate.")
        return

    print(f"Validated {len(results)} pooled placement layout(s):")
    for env_id, episode_index, validation_results in results:
        print(f"env {env_id} episode {episode_index}: {validation_results.report()}")

    num_pass = sum(
        1 for _, _, validation_results in results if validation_results.do_all_required_validation_checks_pass()
    )
    num_settled = sum(
        1
        for _, _, validation_results in results
        if validation_results.validation_results.get(PlacementCheck.PHYSICS_SETTLED)
    )
    print(f"Summary: {num_pass}/{len(results)} pass validation, {num_settled}/{len(results)} physically settled.")
