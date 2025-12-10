# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.utils.pose import Pose

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


class Relation:
    def __init__(self, parent_asset: Asset, child_asset: Asset):
        self.parent_asset: Asset = parent_asset
        self.child_asset: Asset = child_asset

    @abstractmethod
    def resolve(self):
        pass


class OnRelation(Relation):
    def resolve(self):
        # Get the pose of the parent
        # Resolve the pose of the child relative to the parent
        # Add to world frame
        # return the pose in world frame
        print(f"Resolving on relation between {self.parent_asset.name} and {self.child_asset.name}")
        return Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))


class RelationResolver:
    def __init__(self, scene: "Scene"):
        self.scene = scene

    # This would actually set the initial_poses
    def resolve_relations(self):
        # Get assets
        for asset in self.scene.assets.values():
            if isinstance(asset.initial_pose, Relation):
                print(f"Asset {asset.name} has initial pose of type {type(asset.initial_pose)}")
                pose = asset.initial_pose.resolve()
                print(f"Now setting initial pose of asset {asset.name} to {pose}")
                asset.set_initial_pose(pose)
            else:
                print(f"Asset {asset.name} has no initial pose")
