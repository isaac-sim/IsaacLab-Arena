# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter generation from a constructed scene, using Arena relation placement."""

from __future__ import annotations

import math
import torch
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.clutter_geometry import (
    dynamic_rigid_object_keys,
    spawned_geometry_is_fixed,
    spawned_rigid_body_has_gravity,
)
from isaaclab_arena.offline_placement.clutter_params import ClutterSettleParams
from isaaclab_arena.offline_placement.clutter_validation import SettleTracker
from isaaclab_arena.offline_placement.clutter_validators import (
    ClutterPlacementValidator,
    ClutterState,
    RestValidator,
    build_post_physics_validators,
    default_post_physics_validators,
)
from isaaclab_arena.offline_placement.scene_snapshot import SceneSnapshot
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_base_rotation_per_asset, write_layout_to_sim
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validators import build_validators, get_build_time_checks
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


@dataclass(frozen=True)
class ClutterGroup:
    """Rigid objects sharing a fixed clutter support, identified by scene keys."""

    support: str
    """Scene key of the static or kinematic support."""

    objects: tuple[str, ...]
    """Scene keys of the objects to drop together."""


def groups_from_assets(assets: list[PlaceableAsset]) -> list[ClutterGroup]:
    """Read clutter supports and members from their ClutterOn relations."""
    members: dict[str, list[str]] = {}
    for asset in assets:
        relation = get_relation(asset, ClutterOn)
        if relation is not None:
            members.setdefault(relation.parent.get_scene_key(), []).append(asset.get_scene_key())
    assert members, "Environment must declare at least one ClutterOn relation"
    return [ClutterGroup(support, tuple(objects)) for support, objects in members.items()]


@dataclass
class SettledPlacement:
    """One accepted layout and the checks that established its acceptance."""

    poses: dict[str, Pose]
    """Environment-local poses keyed by runtime scene name."""
    validation: dict
    """Pre-physics verdicts, post-physics reports and sampling settings."""


def settle_clutter(
    env: ManagerBasedEnv,
    assets: list[PlaceableAsset],
    *,
    seed: int = 42,
    attempts: int = 5,
    params: ClutterSettleParams | None = None,
    placer_params: ObjectPlacerParams | None = None,
    validators: list[ClutterPlacementValidator] | None = None,
) -> list[SettledPlacement]:
    """Generate one accepted layout per environment and restore the caller's scene state.

    Args:
        env: Constructed scene at the desired initial poses and articulation configuration.
        assets: Scene assets carrying ClutterOn members, IsAnchor supports, and fixed neighbors.
            Object-set variants must be assigned for all environments before scene construction.
        seed: Seed for independent release samples in each environment and attempt.
        attempts: Maximum trials for each environment before failing.
        params: Physics time budget and sampling interval.
        placer_params: Solver, candidate-count and validation settings for release poses.
            None uses fresh ObjectPlacerParams defaults.
        validators: Configured post-physics checks; None enables the default checks.

    Returns:
        Environment-local poses for every dynamic rigid object, indexed by environment.
        Each result includes the effective settings and outcomes of its configured checks.
    """
    env = env.unwrapped
    params = replace(params) if params is not None else ClutterSettleParams()
    placer_params = _release_placer_params(placer_params)
    if validators is None:
        validators = build_post_physics_validators(default_post_physics_validators())
    assert any(validator.enabled for validator in validators), "Enable at least one post-physics validator"
    rest_validators = [
        validator for validator in validators if isinstance(validator, RestValidator) and validator.enabled
    ]
    assert len(rest_validators) <= 1, "At most one enabled RestValidator is supported; configure its thresholds"
    rest_validator = rest_validators[0] if rest_validators else None
    assert attempts > 0, "attempts must be positive"
    groups, placement_assets, collision_objects = _prepare_scene(env, assets)
    capture_keys = dynamic_rigid_object_keys(env.scene)
    geometry_keys = sorted(set(env.scene.rigid_objects) | {group.support for group in groups})
    clutter_keys = {key for group in groups for key in group.objects}
    passive_keys = sorted(set(geometry_keys) - clutter_keys)
    boxes = {key: env.arena_world.get_aabb_in_local_frame(key) for key in geometry_keys}
    snapshot = SceneSnapshot(env, geometry_keys)
    requested_checks = (placer_params.enabled_checks or set()) | (placer_params.required_checks or set())
    requested_checks |= {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION}
    candidate_count = env.num_envs * placer_params.max_placement_attempts
    accepted: dict[int, SettledPlacement] = {}
    failures: dict[int, list[str]] = {i: [] for i in range(env.num_envs)}
    try:
        for attempt in range(attempts):
            snapshot.restore(env)
            pending = [i for i in range(env.num_envs) if i not in accepted]
            placer = ObjectPlacer(replace(placer_params, placement_seed=seed + attempt * candidate_count))
            releases = placer.place_ranked_per_env(
                placement_assets, num_envs=env.num_envs, results_per_env=1, collision_objects=collision_objects
            )
            released, errors = _release_candidates(env, releases, pending, placement_assets, requested_checks)
            trackers = _wait_for_rest(env, capture_keys, released, params, rest_validator)
            for env_id in released:
                measured = {key: env.arena_world.get_pose_e(key)[env_id] for key in capture_keys}
                if not all(torch.isfinite(value).all() for value in measured.values()):
                    errors[env_id] = "non-finite measured pose"
                    continue
                layout = {key: _pose(value) for key, value in measured.items()}
                state = ClutterState(env, env_id, layout, snapshot, passive_keys, groups, boxes, trackers.get(env_id))
                reports = [
                    validator.validate(state) if validator.enabled else validator.report() for validator in validators
                ]
                reasons = [
                    f"{report.check}: {report.reason or 'did not pass'}"
                    for validator, report in zip(validators, reports, strict=True)
                    if validator.enabled and report.passed is not True
                ]
                if reasons:
                    errors[env_id] = "; ".join(reasons)
                else:
                    accepted[env_id] = SettledPlacement(
                        layout,
                        {
                            "pre_physics": dict(releases[env_id][0].validation_results.validation_results),
                            "post_physics": [asdict(report) for report in reports],
                            "sampling": {**asdict(params), "physics_dt_s": env.sim.get_physics_dt()},
                        },
                    )
            for env_id, reason in errors.items():
                failures[env_id].append(f"attempt {attempt + 1}: {reason}")
                print(f"[clutter] env {env_id}, {failures[env_id][-1]}")
            if len(accepted) == env.num_envs:
                return [accepted[i] for i in range(env.num_envs)]
        rejected = {i: failures[i] for i in range(env.num_envs) if i not in accepted}
        raise AssertionError(f"No accepted layout after {attempts} attempt(s): {rejected}")
    finally:
        snapshot.restore(env)


def _prepare_scene(
    env: ManagerBasedEnv, assets: list[PlaceableAsset]
) -> tuple[list[ClutterGroup], list[PlaceableAsset], list[CollisionObject]]:
    """Check support mobility, gravity and placement coverage before any scene writes."""
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects

    for asset in assets:
        if isinstance(asset, RigidObjectSet):
            assignments = asset.variant_indices_by_env
            assert assignments is not None and len(assignments) == env.num_envs, (
                f"Object set {asset.name!r} needs fixed variants for {env.num_envs} environments; "
                f"call assign_variants({env.num_envs}) before building the environment. "
                "Assigning variants after spawning can change the geometry used by the release solver."
            )

    reachability_targets = [asset.get_scene_key() for asset in assets if asset.requires_reachability]
    assert not reachability_targets, f"Offline settling cannot validate final-pose reachability: {reachability_targets}"
    groups = groups_from_assets(assets)
    placement_assets = [asset for asset in assets if asset.get_relations()]
    assert all(
        asset.is_anchor or get_relation(asset, ClutterOn) is not None for asset in placement_assets
    ), "Offline settling requires non-clutter placement to be resolved to fixed anchors first"
    collision_objects = get_passive_collision_objects(assets)
    uncovered = [
        asset.get_scene_key()
        for asset in assets
        if asset.get_scene_key() in env.scene.rigid_objects
        and asset not in placement_assets
        and asset not in collision_objects
    ]
    assert not uncovered, f"Passive rigid objects need fixed poses and collision geometry: {uncovered}"
    members = [key for group in groups for key in group.objects]
    assert len(set(members)) == len(members), "An object must belong to exactly one group"
    assert all(group.support not in members for group in groups), "Supports cannot be clutter members"
    gravity = env.cfg.sim.gravity
    assert gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0, "Offline settling requires downward world-Z gravity"
    for group in groups:
        for env_id, pose in enumerate(env.arena_world.get_pose_e(group.support)):
            try:
                quaternion_to_90_deg_z_quarters(tuple(pose[3:].tolist()))
            except AssertionError as error:
                raise AssertionError(f"Support {group.support!r}, environment {env_id}: {error}") from error
        assert spawned_geometry_is_fixed(
            env.scene, group.support
        ), f"Support {group.support!r} must be static or kinematic"
        for key in group.objects:
            assert key in env.scene.rigid_objects, f"Clutter object {key!r} must be a rigid object"
            assert not spawned_geometry_is_fixed(env.scene, key), f"Clutter object {key!r} must be dynamic"
            assert spawned_rigid_body_has_gravity(env.scene, key), f"Clutter object {key!r} must have gravity enabled"

    return groups, placement_assets, collision_objects


def _release_candidates(
    env: ManagerBasedEnv,
    releases: list[list[PlacementResult]],
    pending: list[int],
    assets: list[PlaceableAsset],
    requested_checks: set[str],
) -> tuple[list[int], dict[int, str]]:
    """Write validated releases for pending environments and report rejected candidates."""
    anchors = {asset for asset in assets if asset.is_anchor}
    released, failures = [], {}
    for env_id in pending:
        release = releases[env_id][0]
        validation = release.validation_results
        missing_checks = requested_checks - validation.validation_results.keys()
        assert not missing_checks, f"Offline release validators did not run: {sorted(missing_checks)}"
        if release.success:
            _release_objects(env, env_id, release, anchors, assets)
            released.append(env_id)
        else:
            failures[env_id] = f"release placement failed: {validation.get_failed_validation_check_names}"
    return released, failures


def _wait_for_rest(
    env: ManagerBasedEnv,
    capture_keys: list[str],
    released: list[int],
    params: ClutterSettleParams,
    rest: RestValidator | None,
) -> dict[int, SettleTracker]:
    """Step the scene until released layouts settle, diverge, or exhaust the time budget."""
    from isaaclab_arena.utils.physics_settle import step_physics

    if not released:
        return {}
    if rest is None:
        step_physics(env, math.floor(params.timeout_s / env.sim.get_physics_dt()))
        return {}
    trackers = {i: SettleTracker(rest) for i in released}
    dt = env.sim.get_physics_dt()
    poll_steps, max_steps = _step_budget(dt, params, rest)
    assert (
        0.5 * abs(env.cfg.sim.gravity[2]) * (poll_steps * dt) ** 2 > rest.move_thresh_m
    ), "Poll interval is too short to distinguish free fall from rest"
    for _ in range(max_steps // poll_steps):
        step_physics(env, poll_steps)
        states = torch.stack([env.arena_world.get_pose_e(key) for key in capture_keys], dim=1)
        for i, tracker in trackers.items():
            if not tracker.diverged:
                tracker.update(states[i, :, :3], states[i, :, 3:])
        if all(tracker.settled or tracker.diverged for tracker in trackers.values()):
            break
    return trackers


def _release_placer_params(params: ObjectPlacerParams | None) -> ObjectPlacerParams:
    """Release settings with mandatory geometry checks."""
    params = params if params is not None else ObjectPlacerParams()
    requested_checks = (params.enabled_checks or set()) | (params.required_checks or set())
    settled_pose_checks = {PlacementCheck.IK_REACHABLE, PlacementCheck.PHYSICS_SETTLED}
    unsupported_checks = requested_checks & settled_pose_checks
    assert (
        not unsupported_checks
    ), f"Offline release validation cannot certify settled-pose checks: {sorted(unsupported_checks)}"
    if params.enabled_checks is not None and params.required_checks is not None:
        assert params.required_checks <= params.enabled_checks, "Required release checks must be enabled"
    geometry_checks = {PlacementCheck.NO_OVERLAP, PlacementCheck.CLUTTER_ON_RELATION}
    enabled_checks = params.enabled_checks
    if enabled_checks is None:
        # Default validation covers available release checks, never settled-pose claims.
        release_checks = set(get_build_time_checks()) - settled_pose_checks
        enabled_checks = {
            validator.check for validator in build_validators(replace(params, enabled_checks=release_checks))
        }
    enabled_checks = enabled_checks | geometry_checks
    required_checks = enabled_checks if params.required_checks is None else params.required_checks | geometry_checks
    return replace(params, enabled_checks=enabled_checks, required_checks=required_checks)


def _release_objects(
    env: ManagerBasedEnv,
    env_id: int,
    layout: PlacementResult,
    anchors: set[PlaceableAsset],
    assets: list[PlaceableAsset],
) -> None:
    """Write a complete ObjectPlacer release layout with zero root velocities."""
    write_layout_to_sim(env, env_id, layout, anchors, get_base_rotation_per_asset(assets))
    env.scene.write_data_to_sim()
    env.sim.forward()


def _pose(value: torch.Tensor) -> Pose:
    """Convert a finite xyz/xyzw tensor of shape (7,) to an Arena pose."""
    assert torch.isfinite(value).all(), "Cannot cache a non-finite pose"
    return Pose(tuple(value[:3].tolist()), tuple(value[3:7].tolist()))


def _step_budget(dt: float, params: ClutterSettleParams, rest: RestValidator) -> tuple[int, int]:
    """Return poll and trial step counts that allow the required quiet windows."""
    poll_steps = math.ceil(params.poll_interval_s / dt)
    max_steps = math.floor(params.timeout_s / dt)
    required_polls = rest.required_quiet_windows + 1
    assert max_steps // poll_steps >= required_polls, (
        f"timeout_s allows {max_steps // poll_steps} polls at physics dt={dt:g}; need {required_polls}. "
        "Increase timeout_s or reduce poll_interval_s."
    )
    return poll_steps, max_steps
