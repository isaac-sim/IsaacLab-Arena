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

from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaac_arena.assets.asset import Asset
from isaac_arena.geometry.pose import Pose


class Background(Asset):
    """
    Encapsulates the background scene config for a environment.
    """

    # Defined in Asset, restated here for clariry
    # name: str | None = None
    # tags: list[str] | None = None
    usd_path: str | None = None
    initial_pose: Pose | None = None
    object_min_z: float | None = None

    def __init__(self):
        super().__init__(self.name)

    def get_cfgs(self) -> dict[str, Any]:
        assert self.name is not None, "Background name is not set"
        background_scene_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            spawn=UsdFileCfg(usd_path=self.usd_path),
        )
        background_scene_cfg = self._add_initial_pose_to_cfg(background_scene_cfg)
        return {
            self.name: background_scene_cfg,
        }

    def _add_initial_pose_to_cfg(self, background_scene_cfg: AssetBaseCfg) -> AssetBaseCfg:
        if self.initial_pose is not None:
            background_scene_cfg.init_state.pos = self.initial_pose.position_xyz
            background_scene_cfg.init_state.rot = self.initial_pose.rotation_wxyz
        return background_scene_cfg
