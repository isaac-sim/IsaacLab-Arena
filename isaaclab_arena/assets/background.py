# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose


class Background(Object):
    """
    Encapsulates the background scene for a environment.
    """

    def __init__(
        self,
        name: str,
        usd_path: str,
        object_min_z: float,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            usd_path=usd_path,
            initial_pose=initial_pose,
            prim_path=prim_path,
            # Backgrounds don't have physics (at the moment)
            object_type=ObjectType.BASE,
            **kwargs,
        )
        # We use this to define reset terms for when objects are dropped.
        # NOTE(alexmillane, 2025.09.19): This is a global z height. If you shift the
        # background, by using initial_pose, this height doesn't shift with it.
        # TODO(alexmillane, 2025.09.19): Make this value relative to the background
        # prim origin.
        self.object_min_z = object_min_z
