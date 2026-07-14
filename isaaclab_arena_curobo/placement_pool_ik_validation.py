# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IK-reachability validation for pooled placement layouts.

Mirrors physics-settle validation: each stored layout is written into the sim and stamped with
an ``IK_REACHABLE`` verdict, here by asking cuRobo whether the robot can reach a top-down grasp
at every movable object. This is a post-hoc check only — it does not feed back into placement.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_events import (
    get_base_rotation_per_object,
    get_movable_object_names,
    get_placement_pool,
    write_layout_to_sim,
)
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena_curobo.curobo_planner_utils import (
    sync_object_poses_in_robot_base_frame,
    top_down_grasp_pose_in_robot_frame,
)
from isaaclab_arena_curobo.ik_utils import check_ik_feasibility_batch_goal_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


def _layout_is_ik_reachable(
    env: ManagerBasedEnv,
    planner,
    movable_object_names: list[str],
    grasp_z_offset: float,
    ik_pos_threshold: float,
    ik_rot_threshold: float,
    env_id: int = 0,
) -> bool:
    """Whether the robot can reach a top-down grasp at every movable object in the current layout.

    Builds every object's grasp pose and checks them in a single batched IK solve against the
    planner's synced collision world. Assumes the layout is already written into ``env_id`` and the
    planner's collision world is synced.
    """
    # TODO(xinjieyao, 2026-06-29): Batch across layouts too (solve_batch_env with per-env collision
    # worlds). SkillGen never tested it; needs extra tweaks in the cuRobo-Lab interface (the lab
    # CuroboPlanner builds a single-env MotionGen/collision world, not a multi-env one).
    if not movable_object_names:
        return True
    grasp_poses = torch.stack(
        [top_down_grasp_pose_in_robot_frame(env, name, grasp_z_offset, env_id) for name in movable_object_names]
    )
    feasible, _, _ = check_ik_feasibility_batch_goal_poses(
        planner,
        grasp_poses,
        position_threshold=ik_pos_threshold,
        rotation_threshold=ik_rot_threshold,
    )
    return bool(feasible.all().item())


def validate_pool_ik(
    env: ManagerBasedEnv,
    planner,
    placement_pool: PooledObjectPlacer | None = None,
    grasp_z_offset: float = 0.02,
    ik_pos_threshold: float = 0.01,
    ik_rot_threshold: float = 0.1,
) -> list[tuple[int, int, PlacementValidationResults]] | None:
    """IK-validate every layout in a placement pool, recording the result on its validation results.

    Args:
        env: The Isaac Lab env; must expose a ``robot`` articulation in its scene.
        planner: A ``CuroboPlanner`` bound to env 0 (see ``make_curobo_planner``).
        placement_pool: PooledObjectPlacer whose stored layouts are validated. When ``None`` it is
            derived from the env's registered pooled layouts.
        grasp_z_offset: Height (m) above each object center for the grasp pose.
        ik_pos_threshold: Max IK position error (m) for a grasp to count as reachable.
        ik_rot_threshold: Max IK rotation error (rad) for a grasp to count as reachable.

    Returns:
        ``(env_id, episode_index, checklist)`` for every layout, in ``(env_id, episode_index)`` order,
        or ``None`` when ``placement_pool`` is omitted and the env has no pooled layouts.
    """
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None

    objects = placement_pool.objects
    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotation_per_object(objects)
    # TODO(xinjieyao, 2026-06-29): Expose the objects-to-reach as an interface
    # rather than checking every movable object. Default it to the task's pickup object(s)
    # then validate only those grasps. For now we batch-check all
    # movable objects.
    movable_object_names = get_movable_object_names(objects, anchor_objects_set)

    layouts_per_env = placement_pool.layouts_per_env()
    num_envs = min(len(layouts_per_env), env.unwrapped.num_envs)

    results: list[tuple[int, int, PlacementValidationResults]] = []
    # The planner is bound to env 0, so every candidate is written into env 0 and checked there.
    for src_env_id in range(num_envs):
        for episode_index, layout in enumerate(layouts_per_env[src_env_id]):
            write_layout_to_sim(env.unwrapped, 0, layout, anchor_objects_set, base_rotations)
            # Sync the object poses into the robot base frame for collision world.
            sync_object_poses_in_robot_base_frame(planner)
            reachable = _layout_is_ik_reachable(
                env.unwrapped,
                planner,
                movable_object_names,
                grasp_z_offset,
                ik_pos_threshold,
                ik_rot_threshold,
            )
            # TODO(xinjieyao, 2026-07-01): Make this persisted verdict reproducible/auditable.
            layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE] = reachable
            results.append((src_env_id, episode_index, layout.validation_results))

    results.sort(key=lambda item: (item[0], item[1]))
    return results


def _stamp_unvalidated_layouts(
    env: ManagerBasedEnv,
    planner,
    placement_pool: PooledObjectPlacer,
    movable_object_names: list[str],
    anchor_objects_set: set,
    base_rotations: dict,
    grasp_z_offset: float,
    ik_pos_threshold: float,
    ik_rot_threshold: float,
) -> None:
    """Stamp ``IK_REACHABLE`` on every stored layout that lacks it, by IK-checking it in the sim.

    The planner is bound to env 0, so each candidate is written into env 0 and checked there. Layouts
    already carrying a verdict are skipped so refill rounds only re-check freshly solved candidates.
    """
    for layouts in placement_pool.layouts_per_env():
        for layout in layouts:
            if PlacementCheck.IK_REACHABLE in layout.validation_results.validation_results:
                continue
            write_layout_to_sim(env.unwrapped, 0, layout, anchor_objects_set, base_rotations)
            sync_object_poses_in_robot_base_frame(planner)
            reachable = _layout_is_ik_reachable(
                env.unwrapped,
                planner,
                movable_object_names,
                grasp_z_offset,
                ik_pos_threshold,
                ik_rot_threshold,
            )
            layout.validation_results.validation_results[PlacementCheck.IK_REACHABLE] = reachable


def filter_pool_by_ik_reachability(
    env: ManagerBasedEnv,
    planner,
    placement_pool: PooledObjectPlacer | None = None,
    target_reachable_per_env: int | None = None,
    grasp_z_offset: float = 0.02,
    ik_pos_threshold: float = 0.01,
    ik_rot_threshold: float = 0.1,
    max_refill_rounds: int = 5,
) -> PooledObjectPlacer | None:
    """Prune a placement pool to IK-reachable layouts, refilling geometry until the target is met.

    Reject-&-refill: every stored layout is IK-checked in the sim and stamped; only layouts whose every
    movable-object top-down grasp is reachable are retained. When an env falls short of
    ``target_reachable_per_env``, fresh geometry layouts are solved and re-checked, up to
    ``max_refill_rounds``. The pool is mutated in place so a later env built from the same config draws
    only reachable layouts. Intended to run on a throwaway env, leaving the eval env's pool untouched.

    Args:
        env: The Isaac Lab env; must expose a ``robot`` articulation in its scene.
        planner: A ``CuroboPlanner`` bound to env 0 (see ``make_curobo_planner``).
        placement_pool: Pool to filter. When ``None`` it is derived from the env's registered layouts.
        target_reachable_per_env: Reachable layouts each env must retain. Defaults to the pool's
            ``min_unique_layouts_per_env`` worth (its per-env stored count at entry).
        grasp_z_offset: Height (m) above each object center for the grasp pose.
        ik_pos_threshold: Max IK position error (m) for a grasp to count as reachable.
        ik_rot_threshold: Max IK rotation error (rad) for a grasp to count as reachable.
        max_refill_rounds: Cap on solve-and-recheck rounds before returning best-effort.

    Returns:
        The filtered pool, or ``None`` when ``placement_pool`` is omitted and the env has no pooled layouts.
    """
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None

    if target_reachable_per_env is None:
        target_reachable_per_env = min((len(layouts) for layouts in placement_pool.layouts_per_env()), default=1)
    target_reachable_per_env = max(1, target_reachable_per_env)

    objects = placement_pool.objects
    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotation_per_object(objects)
    movable_object_names = get_movable_object_names(objects, anchor_objects_set)

    def _is_reachable(layout) -> bool:
        return bool(layout.validation_results.validation_results.get(PlacementCheck.IK_REACHABLE))

    # Each round ends on stamp+retain so the pool never carries unvalidated layouts out of the loop;
    # a refill (which appends unvalidated candidates) only runs when another round will re-check them.
    for round_idx in range(max_refill_rounds):
        _stamp_unvalidated_layouts(
            env,
            planner,
            placement_pool,
            movable_object_names,
            anchor_objects_set,
            base_rotations,
            grasp_z_offset,
            ik_pos_threshold,
            ik_rot_threshold,
        )
        placement_pool.retain_layouts_per_env(_is_reachable)
        reachable_per_env = [len(layouts) for layouts in placement_pool.layouts_per_env()]
        if min(reachable_per_env, default=0) >= target_reachable_per_env:
            print(f"IK filter: reached target {target_reachable_per_env} reachable/env {reachable_per_env}.")
            return placement_pool
        if round_idx == max_refill_rounds - 1:
            break
        try:
            placement_pool.refill_geometry_layouts(target_reachable_per_env)
        except RuntimeError as refill_error:
            print(
                f"IK filter: geometry refill fell short of target ({refill_error}); "
                f"stopping with reachable/env {reachable_per_env}."
            )
            return placement_pool

    print(
        f"IK filter: after {max_refill_rounds} rounds reachable/env "
        f"{[len(layouts) for layouts in placement_pool.layouts_per_env()]}, "
        f"target was {target_reachable_per_env} (best effort)."
    )
    return placement_pool


def print_ik_validation_results(results: list[tuple[int, int, PlacementValidationResults]]) -> None:
    """Print each layout's validation results and an IK-reachable pass/fail summary."""
    if not results:
        print("Placement pool has no layouts to validate.")
        return

    print(f"IK-validated {len(results)} pooled placement layout(s):")
    for env_id, episode_index, validation_results in results:
        print(f"env {env_id} episode {episode_index}: {validation_results.report()}")

    num_reachable = sum(
        1
        for _, _, validation_results in results
        if validation_results.validation_results.get(PlacementCheck.IK_REACHABLE)
    )
    print(f"Summary: {num_reachable}/{len(results)} layouts IK-reachable.")
