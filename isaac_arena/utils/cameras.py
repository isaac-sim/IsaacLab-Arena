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

from copy import deepcopy
from typing import Any

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

# if you already have these utilities in your repo, reuse them
from isaac_arena.utils.configclass import make_configclass


def add_camera_to_environment_cfg(
    scene_cfg: Any,
    enable_cameras: bool,
    tag: str,
):
    """
    Build a configclass instance that adds the selected cameras to the Scene.
    We will also add the observation config if we are enabling cameras.
    camera_defs: mapping like { "agentview_left_camera": {"camera_cfg": TiledCameraCfg(...), "tags": ["teleop"]}, ... }
    """
    if not enable_cameras or not hasattr(scene_cfg, "observation_cameras"):
        return make_configclass("EmptyCamerasSceneCfg", [])(), make_configclass("EmptyCameraObsCfg", [])()

    camera_defs: dict[str, dict[str, Any]] = scene_cfg.observation_cameras
    fields_spec = []
    observation_dict = {}
    for name, meta in camera_defs.items():
        tags: list[str] = meta.get("tags", [])
        # If a tag is provided, only add the camera if it has that tag.
        # If no tag is provided, add all cameras.
        if tag is not None:
            if tag not in tags:
                continue
        cam_cfg = deepcopy(meta["camera_cfg"])
        observation_dict[name] = meta
        # each camera becomes a field on the Scene config
        fields_spec.append((name, type(cam_cfg), cam_cfg))

    if not fields_spec:
        return make_configclass("EmptyCamerasSceneCfg", [])(), make_configclass("EmptyCameraObsCfg", [])()

    CamerasSceneCfg = make_configclass("CamerasSceneCfg", fields_spec)

    return CamerasSceneCfg(), make_camera_observations_cfg(registered_cameras=observation_dict)


def make_camera_observations_cfg(
    registered_cameras: dict[str, dict[str, Any]],
    normalize: bool = False,
):
    """
    Build a configclass instance that adds one ObsTerm per selected camera.
    The SceneEntity name equals the camera field name used in the Scene.
    """

    obs_fields = []
    for name, meta in registered_cameras.items():
        data_type: list[str] = getattr(meta["camera_cfg"], "data_types", ["rgb"])
        term = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg(name), "data_type": data_type, "normalize": normalize},
        )
        # Field name on ObservationsCfg: use the camera name (or add suffix if you like)
        obs_fields.append((name, ObsTerm, term))

    if not obs_fields:
        EmptyCameraObsCfg = make_configclass("EmptyCameraObsCfg", [], bases=(ObsGroup,))
        WrappedEmpty = make_configclass(
            "WrappedCameraObsCfg",
            [("camera_obs", EmptyCameraObsCfg, EmptyCameraObsCfg())],
            namespace={"EmptyCameraObsCfg": EmptyCameraObsCfg},
        )
        return WrappedEmpty()

    # Create the post init to be used in the observation class
    def post_init(self):
        self.enable_corruption = False
        self.concatenate_terms = False

    # Has to inherit from ObsGroup
    AutoCameraObsCfg = make_configclass(
        "AutoCameraObsCfg", obs_fields, bases=(ObsGroup,), namespace={"__post_init__": post_init}
    )

    # Now wrap the observation group in an observation class
    WrappedCameraObsCfg = make_configclass(
        "WrappedCameraObsCfg",
        [("camera_obs", AutoCameraObsCfg, AutoCameraObsCfg())],
        namespace={"AutoCameraObsCfg": AutoCameraObsCfg},
    )

    try:
        AutoCameraObsCfg.__qualname__ = f"{WrappedCameraObsCfg.__name__}.AutoCameraObsCfg"
    except Exception:
        pass

    return WrappedCameraObsCfg()
