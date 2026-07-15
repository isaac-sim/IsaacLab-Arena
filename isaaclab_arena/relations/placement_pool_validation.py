# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.relations.placement_events import (
    get_base_rotation_per_object,
    get_movable_object_names,
    get_placement_pool,
    get_placement_scene_entity_names,
    write_layout_to_sim,
)
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils import physics_settle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_entity import PlacementEntity
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


def _write_layout_to_envs_for_episode_index(
    env: ManagerBasedEnv,
    layouts_per_env: list[list[PlacementResult]],
    num_envs: int,
    episode_index: int,
    anchor_objects_set: set,
    base_rotations: dict[PlacementEntity, tuple[float, float, float, float]],
    scene_entity_names: Mapping[str, str],
) -> list[tuple[int, PlacementResult]]:
    """Write one layout per env for this episode; return the ``(env_id, layout)`` layouts written.

    Envs whose queue is shorter than ``episode_index`` contribute nothing, so the layouts written holds at most one
    entry per env and may be empty on the final episodes.
    """
    layouts_written: list[tuple[int, PlacementResult]] = []
    for env_id in range(num_envs):
        layouts = layouts_per_env[env_id]
        if episode_index < len(layouts):
            layout = layouts[episode_index]
            write_layout_to_sim(
                env.unwrapped,
                env_id,
                layout,
                anchor_objects_set,
                base_rotations,
                scene_entity_names,
            )
            layouts_written.append((env_id, layout))
    return layouts_written


def _compute_physics_settled_and_add_to_validation_results(
    env: ManagerBasedEnv,
    layouts: list[tuple[int, PlacementResult]],
    movable_object_names: list[str],
    settle_params: PhysicsSettleParams,
) -> list[tuple[int, PlacementValidationResults]]:
    """Read back per-object velocities for a list of layouts and stamp ``PHYSICS_SETTLED`` per layout.

    Returns ``(env_id, validation_results)`` per layout; the settle verdict is stamped only if not already present.
    """

    env_ids = [env_id for env_id, _ in layouts]
    settled_per_env = physics_settle.are_all_objects_settled_per_env(
        env, env_ids, movable_object_names, settle_params.lin_vel_thresh, settle_params.ang_vel_thresh
    )
    validation_results_all_envs: list[tuple[int, PlacementValidationResults]] = []
    for (env_id, layout), settled in zip(layouts, settled_per_env):
        validation_results_per_env = layout.validation_results
        if PlacementCheck.PHYSICS_SETTLED not in validation_results_per_env.validation_results:
            validation_results_per_env.add_validation_check(PlacementCheck.PHYSICS_SETTLED, settled)
        validation_results_all_envs.append((env_id, validation_results_per_env))
    return validation_results_all_envs


def validate_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer | None = None,
    settle_params: PhysicsSettleParams | None = None,
    render: bool = False,
    scene_entity_names: Mapping[str, str] | None = None,
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
        scene_entity_names: Isaac Lab scene name for each placement entity.

    Returns:
        ``(env_id, episode_index, checklist)`` for every layout, in ``(env_id, episode_index)`` order,
        or ``None`` when ``placement_pool`` is omitted and the env has no pooled layouts.
    """
    placement_pool_from_event = placement_pool is None
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None
    if settle_params is None:
        settle_params = PhysicsSettleParams()

    objects = placement_pool.objects
    anchor_objects_set = set(get_anchor_objects(objects))
    base_rotations = get_base_rotation_per_object(objects)
    if scene_entity_names is None:
        scene_entity_names = get_placement_scene_entity_names(env)
    if scene_entity_names is None:
        assert not placement_pool_from_event, "Placement reset event is missing scene_entity_names"
        scene_entity_names = {obj.name: obj.name for obj in objects}
    movable_object_names = get_movable_object_names(objects, anchor_objects_set, scene_entity_names)

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

    results: list[tuple[int, int, PlacementValidationResults]] = []
    for episode_index in range(max_episodes):
        # Set layout, then settle and collect results in parallel.
        layouts = _write_layout_to_envs_for_episode_index(
            env,
            layouts_per_env,
            num_envs,
            episode_index,
            anchor_objects_set,
            base_rotations,
            scene_entity_names,
        )
        if layouts:
            physics_settle.step_physics(env, num_physics_steps, render=render)
            validation_results = _compute_physics_settled_and_add_to_validation_results(
                env, layouts, movable_object_names, settle_params
            )
            for env_id, validation_results_per_env in validation_results:
                results.append((env_id, episode_index, validation_results_per_env))
    # The results are in (env_id, episode_index) order, so sort by env_id and then episode_index.
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
