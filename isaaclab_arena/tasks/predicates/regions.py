# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configured full-shape containment predicates for live parent-relative regions."""

from isaaclab_arena.geometry.containment import BoxRegion, RegionContainment


class ObjectInRegion:
    """Cache configured collision geometry while checking measured poses on every call."""

    def __init__(self, cfg, env):
        self.region = BoxRegion(**cfg.params["region"])
        self.geometry = RegionContainment(env.cfg.scene, unit_scale_regions=(self.region.parent_name,))

    def __call__(self, env, object_name: str, region: dict):
        """Return full primitive containment for each environment.

        Args:
            env: Environment exposing configured geometry and live ArenaWorld poses.
            object_name: Rigid component whose supported collision primitives are measured.
            region: Serialized BoxRegion fields; changing them requires a new predicate.

        Returns:
            One Boolean per environment, false for invalid measured frames.
        """
        assert BoxRegion(**region) == self.region, "Region definition changed; construct a new predicate"
        world = env.arena_world
        return self.geometry.measure_region(
            object_name, world.get_pose_w(object_name), world.get_pose_w(self.region.parent_name), self.region
        ).contained
