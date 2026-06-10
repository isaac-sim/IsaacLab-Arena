# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tool that physics-validates every candidate layout in a placement pool and logs the result.

It steps physics on *every* stored candidate so its stability can be inspected ahead of time. The settle outcome is
recorded onto each candidate's ``PlacementValidationChecklist``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_events import (
    get_base_rotations,
    get_movable_object_names,
    get_placement_pool,
    write_layout_to_sim,
)
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils import physics_settle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationChecklist
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


def _write_layouts_per_episode(
    env: ManagerBasedEnv,
    layouts_per_env: list[list[PlacementResult]],
    num_envs: int,
    episode_index: int,
    anchor_objects_set: set,
    base_rotations,
) -> list[tuple[int, PlacementResult]]:
    """Write one layout per env for this episode; return the ``(env_id, layout)`` batch written.

    Envs whose queue is shorter than ``episode_index`` contribute nothing, so the batch holds at most one
    entry per env and may be empty on the final episodes.
    """
    batch: list[tuple[int, PlacementResult]] = []
    for env_id in range(num_envs):
        layouts = layouts_per_env[env_id]
        if episode_index < len(layouts):
            layout = layouts[episode_index]
            write_layout_to_sim(env.unwrapped, env_id, layout, anchor_objects_set, base_rotations)
            batch.append((env_id, layout))
    return batch


def _grade_settled_batch(
    env: ManagerBasedEnv,
    layouts: list[tuple[int, PlacementResult]],
    movable_object_names: list[str],
    settle_params: PhysicsSettleParams,
) -> list[tuple[int, PlacementValidationChecklist]]:
    """Read back per-object velocities for a list of layouts and stamp ``PHYSICS_SETTLED`` per layout.

    Returns ``(env_id, checklist)`` per layout; the settle verdict is stamped only if not already present.
    """

    env_ids = [env_id for env_id, _ in layouts]
    settled_results = physics_settle.objects_settled_per_episode(
        env, env_ids, movable_object_names, settle_params.lin_vel_thresh, settle_params.ang_vel_thresh
    )
    graded_layouts: list[tuple[int, PlacementValidationChecklist]] = []
    for (env_id, layout), settled in zip(layouts, settled_results):
        checklist = layout.validation_checklist
        if PlacementCheck.PHYSICS_SETTLED not in checklist.checklist_items:
            checklist.add_checklist_item(PlacementCheck.PHYSICS_SETTLED, settled)
        graded_layouts.append((env_id, checklist))
    return graded_layouts


def validate_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer | None = None,
    settle_params: PhysicsSettleParams | None = None,
    render: bool = False,
) -> list[tuple[int, int, PlacementValidationChecklist]] | None:
    """Physics-validate every layout in a placement pool, recording the result on its checklist.

    Steps physics on every stored layout and stamps the ``PHYSICS_SETTLED`` outcome onto that
    layout's ``PlacementValidationChecklist``.


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
    # No-ops when no layouts are stored in the pool
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None
    if settle_params is None:
        settle_params = PhysicsSettleParams()

    objects = placement_pool.objects
    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotations(objects, anchor_objects_set)
    movable_object_names = get_movable_object_names(objects, anchor_objects_set)

    # The length of each env queue is controlled by min_unique_layouts_per_env in ObjectPlacerParams.
    layouts_per_env = placement_pool.layouts_per_env()
    # The number of parallel envs SimApp is supposed to run specified by the user
    num_expected_envs = env.unwrapped.num_envs
    # The number of parallel envs that can be run in practice
    num_envs = min(len(layouts_per_env), num_expected_envs)

    # The number of episodes to validate is the length of the longest env queue
    max_episodes = max((len(layouts_per_env[env_id]) for env_id in range(num_envs)), default=0)

    # settle_params.num_steps is in env-step units; convert to physics substeps
    num_physics_steps = settle_params.num_steps * env.unwrapped.cfg.decimation

    results: list[tuple[int, int, PlacementValidationChecklist]] = []
    for episode_index in range(max_episodes):
        # Set layout, then settle and grade them in parallel.
        layouts = _write_layouts_per_episode(
            env, layouts_per_env, num_envs, episode_index, anchor_objects_set, base_rotations
        )
        if layouts:
            physics_settle.step_physics(env, num_physics_steps, render=render)
            graded_layouts = _grade_settled_batch(env, layouts, movable_object_names, settle_params)
            for env_id, checklist in graded_layouts:
                results.append((env_id, episode_index, checklist))

    results.sort(key=lambda item: (item[0], item[1]))
    return results


def log_validation_results(results: list[tuple[int, int, PlacementValidationChecklist]]) -> None:
    """Print each layout's checklist verdict plus a pass/settle summary for a pool validation sweep.

    Uses ``print`` rather than ``logging`` because the SimulationApp reconfigures the root logger on
    launch, which otherwise suppresses module-level log records.
    """
    if not results:
        print("Placement pool has no layouts to validate.")
        return

    print(f"Validated {len(results)} pooled placement layout(s):")
    for env_id, episode_index, checklist in results:
        print(f"  env {env_id} episode {episode_index}: {checklist.report()}")

    num_pass = sum(1 for _, _, checklist in results if checklist.pass_validation_checklist())
    num_settled = sum(1 for _, _, checklist in results if checklist.checklist_items.get(PlacementCheck.PHYSICS_SETTLED))
    print(f"Summary: {num_pass}/{len(results)} pass validation, {num_settled}/{len(results)} physically settled.")
