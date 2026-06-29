# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera config for the LIBERO packing scene's M4 perception bridge.

Adds a fixed exterior camera (the GaP "main"/non-eye_in_hand view) and depth to the wrist camera. The
exterior camera is spawned here and AIMED at runtime via set_world_poses_from_view (see the cap_bridge
scripts), which keeps the cfg minimal and avoids OffsetCfg convention pitfalls. Import this lazily
(inside get_env, after the SimulationApp boots) -- it pulls in Isaac Lab configclasses, which must not
be imported at environment-registration time.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.franka.franka import FrankaCameraCfg

# Camera sensor update_period (seconds). 0.0 = render every step. Set ILA_CAM_UPDATE_PERIOD to render the
# perception cameras less often (e.g. 0.08 = every ~4 steps at 50 Hz) — the per-step render is ~18ms of a
# ~33ms step, and GaP only reads the camera at (static) perceive nodes, so a render cadence is safe.
_CAM_UPDATE_PERIOD = float(os.environ.get("ILA_CAM_UPDATE_PERIOD", "0.0"))

# Fixed top-down exterior camera over the table center. Top-down + cfg pose (no runtime aiming) keeps the
# render, data.pos_w, and pose_mat mutually consistent (verified by the frame-check at 0.000 cm over all
# 6 objects). ROS/OpenCV rot (w,x,y,z)=(0,1,0,0) = Rx(180deg): +Z points down, +X = +X world.
_EXTERIOR_POS = (0.32, 0.0, 1.6)  # top-down over the table top (z=0.40); ~1.2m above the surface
_EXTERIOR_ROT_WXYZ = (0.0, 1.0, 0.0, 0.0)
_EXTERIOR_HW = (512, 800)  # (H, W)
_DEPTH_DT = "distance_to_image_plane"  # perpendicular z-depth, the type GaP back-projection expects


@configclass
class LiberoPerceptionCameraCfg(FrankaCameraCfg):
    """Franka wrist camera (now rgb+depth) plus a fixed exterior rgb+depth camera viewing the table."""

    exterior_cam: CameraCfg = MISSING

    def __post_init__(self):
        # Explicit parent call: @configclass replaces the class object, so zero-arg super() breaks.
        FrankaCameraCfg.__post_init__(self)  # builds wrist_cam with the embodiment's default offset
        self.wrist_cam.data_types = ["rgb", _DEPTH_DT]
        self.wrist_cam.update_period = _CAM_UPDATE_PERIOD

        self.exterior_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/exterior_cam",
            update_period=_CAM_UPDATE_PERIOD,
            height=_EXTERIOR_HW[0],
            width=_EXTERIOR_HW[1],
            data_types=["rgb", _DEPTH_DT],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
            offset=CameraCfg.OffsetCfg(pos=_EXTERIOR_POS, rot=_EXTERIOR_ROT_WXYZ, convention="ros"),
        )
