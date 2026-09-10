# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker
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
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.placement_validation import PlacementValidationResults
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer


def get_pose_capture_assets(assets: list[PlaceableAsset]) -> list[PlaceableAsset]:
    """Return movable non-robot assets."""
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    return [asset for asset in assets if not asset.is_anchor and not isinstance(asset, EmbodimentBase)]


def _write_layout_to_envs_for_episode_index(
    env: ManagerBasedEnv,
    layouts_per_env: list[list[PlacementResult]],
    num_envs: int,
    episode_index: int,
    anchor_assets: set,
    base_rotations: dict[PlaceableAsset, tuple[float, float, float, float]],
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
                anchor_assets,
                base_rotations,
            )
            layouts_written.append((env_id, layout))
    return layouts_written


def _compute_physics_settled_and_add_to_validation_results(
    env: ManagerBasedEnv,
    layouts: list[tuple[int, PlacementResult]],
    movable_object_names: list[str],
    settle_params: PhysicsSettleParams,
    settled_per_env_override: list[bool] | None = None,
) -> list[tuple[int, PlacementValidationResults]]:
    """Update each layout's physics-settled verdict and return its validation results."""

    env_ids = [env_id for env_id, _ in layouts]
    if settled_per_env_override is not None:
        settled_per_env = [settled_per_env_override[env_id] for env_id in env_ids]
    else:
        settled_per_env = physics_settle.are_all_objects_settled_per_env(
            env, env_ids, movable_object_names, settle_params.lin_vel_thresh, settle_params.ang_vel_thresh
        )
    validation_results_all_envs: list[tuple[int, PlacementValidationResults]] = []
    for (env_id, layout), settled in zip(layouts, settled_per_env):
        validation_results_per_env = layout.validation_results
        validation_results_per_env.validation_results[PlacementCheck.PHYSICS_SETTLED] = settled
        validation_results_all_envs.append((env_id, validation_results_per_env))
    return validation_results_all_envs


def _capture_settled_poses_into_layouts(
    env: ManagerBasedEnv,
    layouts: list[tuple[int, PlacementResult]],
    assets: list[PlaceableAsset],
) -> None:
    """Record finite resting positions and full quaternions in each layout."""
    scene = env.unwrapped.scene
    env_origins = scene.env_origins
    for env_id, layout in layouts:
        for asset in assets:
            root_state = wp.to_torch(scene[asset.get_scene_key()].data.root_link_pose_w)[env_id]
            assert torch.isfinite(root_state).all(), f"Non-finite settled pose for {asset.name!r} in env {env_id}"
            position = root_state[:3] - env_origins[env_id]
            rotation = root_state[3:7]
            layout.positions[asset] = (float(position[0]), float(position[1]), float(position[2]))
            layout.rotations[asset] = (
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
                float(rotation[3]),
            )


def _step_until_poses_are_quiet(
    env: ManagerBasedEnv,
    movable_object_names: list[str],
    max_physics_steps: int,
    params: ClutterSettleParams,
    poll_every: int,
    render: bool = False,
    subset_indices: list[int] | None = None,
) -> tuple[list[bool], list[bool]]:
    """Step physics until all environments are quiet or the step budget expires.

    Args:
        env: Simulation environment.
        movable_object_names: N scene keys to monitor.
        max_physics_steps: Maximum physics steps.
        params: Rest thresholds.
        poll_every: Steps per pose sample.
        render: Whether to render each step.
        subset_indices: Object indices whose poses will be captured; None selects all N.

    Returns:
        All-object and captured-object rest flags, each with length env.num_envs.
    """
    # A free-falling object must move farther than the quiet threshold between polls.
    physics_dt = env.unwrapped.sim.get_physics_dt()
    gravity_magnitude = float(torch.linalg.vector_norm(torch.tensor(env.unwrapped.cfg.sim.gravity)))
    free_fall = 0.5 * gravity_magnitude * (poll_every * physics_dt) ** 2
    assert free_fall > params.move_thresh_m, (
        f"poll_every={poll_every} is too frequent to detect motion: a free-falling object moves "
        f"{free_fall * 1000:.2f} mm between polls, under the {params.move_thresh_m * 1000:.2f} mm "
        "movement threshold, so a falling pile would be reported as settled."
    )

    scene = env.unwrapped.scene
    num_envs = env.unwrapped.num_envs
    trackers = [SettleTracker(params) for _ in range(num_envs)]
    subset_trackers = [SettleTracker(params) for _ in range(num_envs)]
    stepped = 0
    while stepped < max_physics_steps:
        # A partial poll could misclassify motion as quiet.
        if max_physics_steps - stepped < poll_every:
            break
        physics_settle.step_physics(env, poll_every, render=render)
        stepped += poll_every
        states = torch.stack([wp.to_torch(scene[name].data.root_link_pose_w) for name in movable_object_names], dim=1)
        settled = [
            tracker.update(states[env_id, :, :3], states[env_id, :, 3:7]) for env_id, tracker in enumerate(trackers)
        ]
        subset_settled = (
            settled
            if subset_indices is None
            else [
                tracker.update(states[env_id][subset_indices, :3], states[env_id][subset_indices, 3:7])
                for env_id, tracker in enumerate(subset_trackers)
            ]
        )
        if all(settled) and all(subset_settled):
            return settled, subset_settled
    if subset_indices is None:
        settled = [tracker.settled for tracker in trackers]
        return settled, settled
    return [tracker.settled for tracker in trackers], [tracker.settled for tracker in subset_trackers]


def validate_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer | None = None,
    settle_params: PhysicsSettleParams | None = None,
    render: bool = False,
    capture_settled_poses: bool = False,
    pose_settle_params: ClutterSettleParams | None = None,
    poll_every: int = 50,
    validate_captured_layout: Callable[[int, PlacementResult], None] | None = None,
) -> list[tuple[int, int, PlacementValidationResults]] | None:
    """Check pooled layouts in physics and optionally record resting poses.

    Args:
        env: Simulation environment.
        placement_pool: Layout pool; defaults to the registered reset pool.
        settle_params: Step budget and velocity thresholds.
        render: Whether to render physics steps.
        capture_settled_poses: Record resting poses and restore scene state after the sweep.
        pose_settle_params: Pose-window thresholds; required for pose capture.
        poll_every: Physics steps per pose sample.
        validate_captured_layout: Callback (env_id, layout) while the captured scene is live.

    Returns:
        (env_id, layout_index, validation_results) for each layout, or None without a pool.
    """
    if placement_pool is None:
        placement_pool = get_placement_pool(env)
        if placement_pool is None:
            return None
    if settle_params is None:
        settle_params = PhysicsSettleParams()

    assert not capture_settled_poses or pose_settle_params is not None, "Pose capture requires a rest check"
    assets = placement_pool.objects
    anchor_assets = set(get_anchor_objects(assets))
    # Capture displaced neighbours too; the driven embodiment retains its configured reset state.
    capture_assets = get_pose_capture_assets(assets)
    base_rotations = get_base_rotation_per_asset(assets)
    movable_object_names = get_movable_asset_names(assets, anchor_assets)
    # Track captured objects separately from driven bodies.
    capture_keys = {asset.get_scene_key() for asset in capture_assets}
    capture_indices = [index for index, name in enumerate(movable_object_names) if name in capture_keys]

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

    initial_state = env.unwrapped.scene.get_state() if capture_settled_poses else None
    try:
        results: list[tuple[int, int, PlacementValidationResults]] = []
        for episode_index in range(max_episodes):
            if capture_settled_poses:
                env.unwrapped.scene.reset_to(initial_state)
                env.unwrapped.sim.forward()
            # Set layout, then settle and collect results in parallel.
            layouts = _write_layout_to_envs_for_episode_index(
                env,
                layouts_per_env,
                num_envs,
                episode_index,
                anchor_assets,
                base_rotations,
            )
            if layouts:
                settled_per_env_override = None
                captured_per_env = None
                if pose_settle_params is None:
                    physics_settle.step_physics(env, num_physics_steps, render=render)
                else:
                    settled_per_env_override, captured_per_env = _step_until_poses_are_quiet(
                        env,
                        movable_object_names,
                        num_physics_steps,
                        pose_settle_params,
                        poll_every=poll_every,
                        render=render,
                        subset_indices=capture_indices,
                    )
                if captured_per_env is not None:
                    for env_id, layout in layouts:
                        layout.validation_results.validation_results[PlacementCheck.CAPTURED_OBJECTS_SETTLED] = (
                            captured_per_env[env_id]
                        )
                if capture_settled_poses:
                    # Never cache poses from a layout that exhausted its budget while moving.
                    settled_layouts = (
                        layouts
                        if captured_per_env is None
                        else [(env_id, layout) for env_id, layout in layouts if captured_per_env[env_id]]
                    )
                    _capture_settled_poses_into_layouts(env, settled_layouts, capture_assets)
                    if validate_captured_layout is not None:
                        for env_id, layout in settled_layouts:
                            validate_captured_layout(env_id, layout)
                validation_results = _compute_physics_settled_and_add_to_validation_results(
                    env, layouts, movable_object_names, settle_params, settled_per_env_override
                )
                for env_id, validation_results_per_env in validation_results:
                    results.append((env_id, episode_index, validation_results_per_env))
    finally:
        if initial_state is not None:
            env.unwrapped.scene.reset_to(initial_state)
            env.unwrapped.sim.forward()
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
