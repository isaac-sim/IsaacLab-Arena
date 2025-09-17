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

import inspect
from contextlib import suppress
from dataclasses import fields, is_dataclass
from typing import Any

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg  # noqa: F401

from isaac_arena.utils.configclass import make_configclass


def make_camera_observation_cfg(
    camera_cfg: Any,
    normalize: bool = False,
):
    """
    Build a configclass instance that adds one ObsTerm per selected camera.
    The SceneEntity name equals the camera field name plus the data type used in the Scene.
    For example, if the camera field name is "robot_pov_cam" and the data type is "rgb", the SceneEntity name will be "robot_pov_cam_rgb".
    We create a class which has a member pointing to another class which is based on the ObsGroup class.
    """

    # If they passed the class, instantiate it so we can read values
    if inspect.isclass(camera_cfg):
        camera_cfg = camera_cfg()

    if not is_dataclass(camera_cfg):
        raise TypeError("camera_cfg must be a dataclass/configclass class or instance")

    obs_fields = []
    for f in fields(camera_cfg):
        name = f.name
        cam = getattr(camera_cfg, name)
        # Skip non-camera fields
        if not isinstance(cam, CameraCfg):
            continue

        # Get modalities from the camera cfg (fallback to rgb)
        dtypes = getattr(cam, "data_types", None) or ["rgb"]
        # one ObsTerm per modality
        for dt in dtypes:
            field_name = f"{name}_{dt}"
            term = ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg(name), "data_type": dt, "normalize": normalize},
            )
            # Field name on ObservationsCfg: use the camera name (or add suffix if you like)
            obs_fields.append((field_name, ObsTerm, term))

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

    with suppress(Exception):
        AutoCameraObsCfg.__qualname__ = f"{WrappedCameraObsCfg.__name__}.AutoCameraObsCfg"

    return WrappedCameraObsCfg()
