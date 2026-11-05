# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena wrappers for the generated laptop and wooden table."""

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose


class Laptop(Object, Openable):
    """Expose the laptop's passive lid hinge through Arena's Openable affordance."""

    def __init__(self, usd_path: str, initial_pose: Pose):
        super().__init__(
            name="laptop",
            usd_path=usd_path,
            object_type=ObjectType.ARTICULATION,
            initial_pose=initial_pose,
            openable_joint_name="lid_hinge",
            openable_threshold=0.75,
        )


class WoodenTable(Object):
    """Expose the kinematic tabletop as a contact target and drop-height reference."""

    def __init__(self, usd_path: str, initial_pose: Pose):
        super().__init__(
            name="table",
            usd_path=usd_path,
            object_type=ObjectType.RIGID,
            initial_pose=initial_pose,
        )
        self.object_min_z = initial_pose.position_xyz[2] - 0.20
