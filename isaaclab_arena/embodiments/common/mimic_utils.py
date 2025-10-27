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

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils


def get_rigid_and_articulated_object_poses(state: dict, env_ids: Sequence[int] | None = None):
    """
    Gets the pose of each object(rigid and articulated) in the current scene.
    Args:
        state: The state of the scene.
    Returns:
        A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
    """

    def pose_from(obj_state) -> "torch.Tensor":
        rp = obj_state["root_pose"][env_ids]  # (..., 7): [x,y,z, qx,qy,qz,qw]
        pos, quat = rp[..., :3], rp[..., 3:7]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))  # (..., 4, 4)

    groups = ["rigid_object", "articulation"]

    object_pose_matrix = {
        name: pose_from(obj_state) for group in groups for name, obj_state in state.get(group, {}).items()
    }

    return object_pose_matrix
