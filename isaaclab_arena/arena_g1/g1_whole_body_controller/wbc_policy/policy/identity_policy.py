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

import numpy as np

from isaaclab_arena.arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy


class IdentityPolicy(WBCPolicy):
    """Identity policy that passes through the target pose."""

    def __init__(self):
        """Initialize the identity policy."""
        self.reset()

    def get_action(self, target_pose: np.ndarray) -> dict[str, np.ndarray]:
        """Get the action for the identity policy."""
        return {"q": target_pose}

    def reset(self):
        pass
