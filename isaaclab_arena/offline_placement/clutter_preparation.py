# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check clutter scene prerequisites before the shared placement collection workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.clutter_geometry import (
    assert_support_reference_transform,
    fixed_poses_match,
    spawned_geometry_is_fixed,
    spawned_rigid_body_has_gravity,
    spawned_rigid_body_is_dynamic,
)
from isaaclab_arena.relations.relations import ClutterOn, get_relation
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def prepare_clutter_settling(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Check support mobility, gravity and placement coverage before any scene writes."""
    clutter_assets = [asset for asset in assets if get_relation(asset, ClutterOn) is not None]
    if not clutter_assets:
        return

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
    from isaaclab_arena.relations.passive_collision_objects import discover_passive_assets

    # Reusable root poses do not identify which object-set variant was spawned.
    assert not has_heterogeneous_objects(assets), "Resolve object sets before collecting clutter layouts"
    reachability_targets = [asset.get_scene_key() for asset in clutter_assets if asset.requires_reachability]
    assert not reachability_targets, f"Cannot validate reachability after clutter drops: {reachability_targets}"

    placement_assets = [asset for asset in assets if asset.get_relations()]
    assert all(
        asset.is_anchor or asset in clutter_assets for asset in placement_assets
    ), "Offline settling requires non-clutter placement to be resolved to fixed anchors first"
    # Inspect original assets, before solver collision discovery aggregates their geometry.
    passive_assets = discover_passive_assets(assets)
    uncovered = [
        asset.get_scene_key()
        for asset in assets
        if asset.get_scene_key() in env.scene.rigid_objects
        and asset not in placement_assets
        and asset not in passive_assets
    ]
    assert not uncovered, f"Passive rigid objects need fixed poses and collision geometry: {uncovered}"

    gravity = env.cfg.sim.gravity
    assert gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0, "Offline settling requires downward world-Z gravity"
    support_keys = set()
    for asset in clutter_assets:
        support_keys.add(get_relation(asset, ClutterOn).parent.get_scene_key())
        key = asset.get_scene_key()
        assert key in env.scene.rigid_objects, f"Clutter object {key!r} must be a rigid object"
        assert spawned_rigid_body_is_dynamic(env.scene, key), f"Clutter object {key!r} must be dynamic"
        assert spawned_rigid_body_has_gravity(env.scene, key), f"Clutter object {key!r} must have gravity enabled"

    for key in sorted(support_keys):
        assert spawned_geometry_is_fixed(env.scene, key), f"Support {key!r} must be static or kinematic"
        assert_support_reference_transform(env.scene, key)
        for pose in env.arena_world.get_pose_e(key):
            # This conversion also rejects tilt or yaw outside the supported quarter turns.
            quaternion_to_90_deg_z_quarters(tuple(pose[3:].tolist()))

    fixed_assets = [
        asset for asset in assets if asset.is_anchor or asset in passive_assets or isinstance(asset, Background)
    ]
    # Static BASE and background transforms are not restored by ordinary root-pose reset events.
    _check_fixed_scene_poses(env, fixed_assets)


def _check_fixed_scene_poses(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> None:
    """Reject live transforms that differ from the fixed geometry used by the solver."""
    for asset in assets:
        pose = asset.get_initial_pose() or Pose.identity()
        assert isinstance(pose, Pose), f"Fixed scene asset {asset.name!r} requires a fixed Pose"
        current = env.arena_world.get_pose_e(asset.get_scene_key())
        expected = pose.to_tensor(device=current.device).expand_as(current)
        assert fixed_poses_match(expected, current), (
            f"Fixed scene asset {asset.get_scene_key()!r} differs from its configured pose. "
            "Reset or rebuild the scene at its configured poses before settling."
        )
