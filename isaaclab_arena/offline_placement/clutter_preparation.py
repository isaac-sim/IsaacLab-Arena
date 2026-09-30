# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check clutter scene prerequisites before the shared placement collection workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.clutter_geometry import (
    assert_flat_support_surface,
    spawned_geometry_is_fixed,
    spawned_rigid_body_has_gravity,
    spawned_rigid_body_is_dynamic,
)
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.physics_settle import get_pose_drift
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def prepare_clutter_settling(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Check support mobility, gravity and placement coverage before any scene writes."""
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.passive_collision_objects import discover_passive_assets

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
    relations = [get_relation(asset, ClutterOn) for asset in assets if get_relation(asset, ClutterOn) is not None]
    assert relations, "Environment must declare at least one ClutterOn relation"
    placement_assets = [asset for asset in assets if asset.get_relations()]
    assert all(
        asset.is_anchor or get_relation(asset, ClutterOn) is not None for asset in placement_assets
    ), "Offline settling requires non-clutter placement to be resolved to fixed anchors first"
    # Check source assets before MESH discovery aggregates their collision geometry.
    passive_assets = discover_passive_assets(assets)
    uncovered = [
        asset.get_scene_key()
        for asset in assets
        if asset.get_scene_key() in env.scene.rigid_objects
        and asset not in placement_assets
        and asset not in passive_assets
    ]
    assert not uncovered, f"Passive rigid objects need fixed poses and collision geometry: {uncovered}"
    fixed_assets = [
        asset for asset in assets if asset.is_anchor or asset in passive_assets or isinstance(asset, Background)
    ]
    gravity = env.cfg.sim.gravity
    assert gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0, "Offline settling requires downward world-Z gravity"
    for asset in placement_assets:
        relation = get_relation(asset, ClutterOn)
        if relation is None:
            continue
        support_key = relation.parent.get_scene_key()
        assert spawned_geometry_is_fixed(env.scene, support_key), f"Support {support_key!r} must be static or kinematic"
        key = asset.get_scene_key()
        assert key in env.scene.rigid_objects, f"Clutter object {key!r} must be a rigid object"
        assert spawned_rigid_body_is_dynamic(
            env.scene, key
        ), f"Clutter object {key!r}: every spawned variant must be dynamic"
        assert spawned_rigid_body_has_gravity(env.scene, key), f"Clutter object {key!r} must have gravity enabled"

    support_keys = {relation.parent.get_scene_key() for relation in relations}
    for support_key in sorted(support_keys):
        assert_flat_support_surface(env.scene, support_key)
    _check_fixed_scene_poses(env, fixed_assets)
    for support_key in sorted(support_keys):
        for pose in env.arena_world.get_pose_e(support_key):
            quaternion_to_90_deg_z_quarters(tuple(pose[3:].tolist()))


def _check_fixed_scene_poses(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Reject live transforms that differ from the fixed geometry used by the solver."""
    for asset in assets:
        pose = asset.get_initial_pose()
        if pose is None:
            pose = Pose.identity()
        assert isinstance(pose, Pose), f"Fixed scene asset {asset.name!r} requires a fixed Pose"
        current = env.arena_world.get_pose_e(asset.get_scene_key())
        expected = pose.to_tensor(device=current.device).expand_as(current)
        drift = get_pose_drift(expected, current)
        # Allow float32 transform roundoff, not physical movement of fixed geometry.
        assert drift is not None and drift[0] <= 1e-5 and drift[1] <= 1e-3, (
            f"Fixed scene asset {asset.get_scene_key()!r} differs from its configured pose. "
            "Reset or rebuild the scene at its configured poses before settling."
        )
