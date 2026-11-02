# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure the editorial camera independently of the policy's image inputs."""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg


def make_camera():
    settings = json.loads(Path(__file__).with_name("camera.json").read_text())
    t_W_C = np.array(settings["eye"])
    forward_W = np.array(settings["target"]) - t_W_C
    forward_W /= np.linalg.norm(forward_W)
    left_W = np.cross((0, 0, 1), forward_W)
    left_W /= np.linalg.norm(left_W)
    up_W = np.cross(forward_W, left_W)
    q_W_C = Rotation.from_matrix(np.column_stack((forward_W, left_W, up_W))).as_quat()
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/FilmCamera",
        update_period=0.0,
        width=1200,
        height=900,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=settings["focal_length"],
            horizontal_aperture=20.955,
            vertical_aperture=15.71625,
            clipping_range=(0.05, 30),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(t_W_C), rot=tuple(q_W_C), convention="world"),
    )
