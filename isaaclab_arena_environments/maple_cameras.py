# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera config for the Maple-table scene's GaP perception bridge.

Adds the fixed exterior RGB-D camera that the GaP DROID adapter reads (the non-eye_in_hand "main" view),
mirroring isaaclab_arena_environments/libero_cameras.py but positioned for the Maple-table geometry.

Contract (Arena-owned, live-pose): Arena owns the camera pose. The camera is given a STATIC agentview
OffsetCfg, so Isaac populates cam.data.pos_w / quat_w_ros / intrinsic_matrices correctly and the
cap-implementer's adapter consumes those LIVE each frame (no re-aim, no cached pose_mat — a runtime
set_world_poses_from_view would both clobber reset-time variation and leave cam.data.pos_w stale). The
pose is agentview, not top-down (top-down fails open-vocab VLM recognition), and is derived from eye/target
via the same create_rotation_matrix_from_view + diag(1,-1,-1) flip the adapter uses to build T_wc, so the
published quat_w_ros matches what GaP back-projection expects.

Workspace/tabletop geometry is NOT derived from object_min_z (that is the task drop/failure threshold, not
the surface height). The eye/target below were validated against the rendered exterior_cam framing of the
built scene (via the staging asset override): the table and pick/destination/distractor objects are well
framed and separated for open-vocab perception. They may still be fine-tuned (e.g. to center the objects).

Import lazily (inside get_env, after the SimulationApp boots): it pulls in Isaac Lab configclasses and
math, which must not be imported at environment-registration time.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

# Camera sensor update_period (seconds). 0.0 = render every step. Set ILA_CAM_UPDATE_PERIOD to render the
# perception camera less often; GaP only reads it at (static) perceive nodes, so a render cadence is safe.
_CAM_UPDATE_PERIOD = float(os.environ.get("ILA_CAM_UPDATE_PERIOD", "0.0"))

# Tight oblique view over the relation-solved workspace. Keeping the Maple table's outer silhouette outside
# the frame matters for the unchanged generic GaP graph: its containment NMS otherwise treats the full table
# as a parent detection and removes the smaller grocery boxes. This remains oblique enough to preserve the
# product labels/shapes used by open-vocabulary perception; it is not a top-down view.
_CAM_EYE = (0.95, 0.0, 0.90)
_CAM_TARGET = (0.42, 0.03, 0.05)
_CAM_FOCAL_LENGTH = 7.0
_EXTERIOR_HW = (512, 800)  # (H, W)
_DEPTH_DT = "distance_to_image_plane"  # perpendicular z-depth, the type GaP back-projection expects


def _agentview_ros_quat_xyzw(eye, target):
    """Camera world orientation as an ``(x, y, z, w)`` quaternion for a static ``OffsetCfg(convention="ros")``.

    This installed Isaac Lab uses xyzw order for both ``quat_from_matrix`` and ``CameraCfg.OffsetCfg.rot``
    (identity == (0, 0, 0, 1)). The rotation is R_cam_to_world = ``create_rotation_matrix_from_view(eye,
    target, "Z") @ diag(1, -1, -1)`` (OpenGL -Z-forward -> OpenCV/ROS-optical +Z-forward), matching the
    ``cam.data.quat_w_ros`` the live-pose adapter reads.
    """
    import torch

    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    eye_t = torch.tensor([eye], dtype=torch.float32)
    tgt_t = torch.tensor([target], dtype=torch.float32)
    r_cam_to_world = create_rotation_matrix_from_view(eye_t, tgt_t, "Z")[0] @ torch.diag(
        torch.tensor([1.0, -1.0, -1.0])
    )
    return tuple(quat_from_matrix(r_cam_to_world).tolist())  # xyzw, as OffsetCfg.rot expects


@configclass
class MapleDroidPerceptionCameraCfg(DroidCameraCfg):
    """Droid (Robotiq) cameras plus a fixed exterior rgb+depth agentview camera the GaP bridge reads.

    Subclasses DroidCameraCfg (whose wrist/external cams mount on links that exist on the Robotiq USD) and
    adds a static-pose exterior_cam aimed at the Maple workspace; the cap-implementer's adapter reads its
    live cam.data.pos_w / quat_w_ros / intrinsics.
    """

    exterior_cam: CameraCfg = MISSING

    def __post_init__(self):
        DroidCameraCfg.__post_init__(self)
        self.exterior_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/exterior_cam",
            update_period=_CAM_UPDATE_PERIOD,
            update_latest_camera_pose=True,
            height=_EXTERIOR_HW[0],
            width=_EXTERIOR_HW[1],
            data_types=["rgb", _DEPTH_DT],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_CAM_FOCAL_LENGTH,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_CAM_EYE, rot=_agentview_ros_quat_xyzw(_CAM_EYE, _CAM_TARGET), convention="ros"
            ),
        )
