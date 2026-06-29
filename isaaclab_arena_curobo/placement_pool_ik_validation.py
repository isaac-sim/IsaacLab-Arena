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

from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_events import (
    get_base_rotation_per_object,
    get_movable_object_names,
    get_placement_pool,
    write_layout_to_sim,
)
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils import physics_settle
from isaaclab_arena_curobo.curobo_planner_utils import top_down_grasp_pose_in_robot_frame
from isaaclab_arena_curobo.ik_utils import check_ik_feasibility

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

    Assumes the layout is already written into ``env_id`` and the planner's collision world is synced.
    Returns False as soon as one object is unreachable.
    """
    reachable = True
    for object_name in movable_object_names:
        grasp_pose = top_down_grasp_pose_in_robot_frame(env, object_name, grasp_z_offset, env_id)
        feasible, _, _, _ = check_ik_feasibility(
            planner,
            grasp_pose,
            position_threshold=ik_pos_threshold,
            rotation_threshold=ik_rot_threshold,
        )
        reachable = reachable and feasible
    return reachable


def validate_pool_ik(
    env: ManagerBasedEnv,
    planner,
    placement_pool: PooledObjectPlacer | None = None,
    settle_steps: int = 5,
    grasp_z_offset: float = 0.02,
    ik_pos_threshold: float = 0.01,
    ik_rot_threshold: float = 0.1,
    render: bool = False,
) -> list[tuple[int, int, PlacementValidationResults]] | None:
    """IK-validate every layout in a placement pool, recording the result on its validation results.

    Each layout is written into env 0 (the planner's env), briefly settled so the sim buffers and
    the planner's collision world reflect the placed poses, then checked for top-down grasp
    reachability at every movable object. The ``IK_REACHABLE`` outcome is stamped onto that layout's
    ``PlacementValidationResults``.

    Args:
        env: The Isaac Lab env; must expose a ``robot`` articulation in its scene.
        planner: A ``CuroboPlanner`` bound to env 0 (see ``make_curobo_planner``).
        placement_pool: PooledObjectPlacer whose stored layouts are validated. When ``None`` it is
            derived from the env's registered pooled layouts.
        settle_steps: Environment steps to advance after writing each layout so the sim and planner
            world reflect it. Converted to physics substeps internally (x the env's decimation).
        grasp_z_offset: Height (m) above each object center for the grasp pose.
        ik_pos_threshold: Max IK position error (m) for a grasp to count as reachable.
        ik_rot_threshold: Max IK rotation error (rad) for a grasp to count as reachable.
        render: When True, render each settle step so the sweep is visible in the GUI. Defaults to False.

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
    movable_object_names = get_movable_object_names(objects, anchor_objects_set)

    layouts_per_env = placement_pool.layouts_per_env()
    num_envs = min(len(layouts_per_env), env.unwrapped.num_envs)
    num_physics_steps = settle_steps * env.unwrapped.cfg.decimation

    results: list[tuple[int, int, PlacementValidationResults]] = []
    # The planner is bound to env 0, so every candidate is written into env 0 and checked there.
    for src_env_id in range(num_envs):
        for episode_index, layout in enumerate(layouts_per_env[src_env_id]):
            write_layout_to_sim(env.unwrapped, 0, layout, anchor_objects_set, base_rotations)
            physics_settle.step_physics(env, num_physics_steps, render=render)
            planner._sync_object_poses_with_isaaclab()
            reachable = _layout_is_ik_reachable(
                env.unwrapped,
                planner,
                movable_object_names,
                grasp_z_offset,
                ik_pos_threshold,
                ik_rot_threshold,
            )
            if PlacementCheck.IK_REACHABLE not in layout.validation_results.validation_results:
                layout.validation_results.add_validation_check(PlacementCheck.IK_REACHABLE, reachable)
            results.append((src_env_id, episode_index, layout.validation_results))

    results.sort(key=lambda item: (item[0], item[1]))
    return results


def print_ik_validation_results(results: list[tuple[int, int, PlacementValidationResults]]) -> None:
    """Print each layout's validation results and an IK-reachable pass/fail summary."""
    if not results:
        print("Placement pool has no layouts to validate.")
        return

    print(f"IK-validated {len(results)} pooled placement layout(s):")
    for env_id, episode_index, validation_results in results:
        print(f"env {env_id} episode {episode_index}: {validation_results.report()}")

    num_reachable = sum(
        1 for _, _, validation_results in results if validation_results.validation_results.get(PlacementCheck.IK_REACHABLE)
    )
    print(f"Summary: {num_reachable}/{len(results)} layouts IK-reachable.")
