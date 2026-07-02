# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""M4 frame-check: validate the exterior-camera pose_mat convention (review R5) before full perception.

The make-or-break for grasping: GaP back-projects masks with cam["pose"] (a 4x4 mapping OpenCV
camera-frame points -> robot-BASE frame) and plans in the base frame. ILA's base is at world
(-0.20,0,0). We replicate GaP's exact back-projection (X=(u-cx)Z/fx, Y=(v-cy)Z/fy, p=pose_mat@[X,Y,Z,1])
and confirm a known object reconstructs to its ground-truth base-frame position to < 1 cm.

pose_mat = T_world_base^-1 @ T_world_cam(quat_w_ros). No optical flip (Isaac quat_w_ros is already
OpenCV). Built with Isaac's own frame utils so quaternion order is handled consistently.

Run:
    ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m4_framecheck.py \
        --headless --num_envs 1 --enable_cameras --placement_seed 0 libero_object_packing
"""

from __future__ import annotations

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_CAM = "exterior_cam"
_DEPTH_DT = "distance_to_image_plane"
_SETTLE_STEPS = 25
_TOL_M = 0.01


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        env = get_arena_builder_from_cli(args_cli).make_registered()
        env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        action = torch.zeros(env.action_space.shape, device=device)
        cam = unwrapped.scene[_CAM]
        robot = unwrapped.scene["robot"]
        for _ in range(_SETTLE_STEPS):
            env.step(action)

        cam_pos = cam.data.pos_w[0:1]  # (1,3) world
        cam_quat = cam.data.quat_w_ros[0:1]  # (1,4) wxyz, OpenCV optical
        K = cam.data.intrinsic_matrices[0].cpu().numpy()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        rgb = cam.data.output["rgb"][0].float()
        depth = cam.data.output[_DEPTH_DT][0].squeeze(-1).cpu().numpy()  # (H,W) meters, z-depth
        H, W = depth.shape

        base_pos = robot.data.root_pos_w[0:1]
        base_quat = robot.data.root_quat_w[0:1]
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        # pose_mat = T_base_cam (camera-to-base, OpenCV). Built from Isaac's own frame utils so quat order
        # is internally consistent (no manual wxyz/xyzw juggling).
        pos_cb, quat_cb = subtract_frame_transforms(base_pos, base_quat, cam_pos, cam_quat)
        pose_mat = torch.eye(4, device=device)
        pose_mat[:3, :3] = matrix_from_quat(quat_cb)[0]
        pose_mat[:3, 3] = pos_cb[0]
        pose_mat_np = pose_mat.cpu().numpy()

        print(f"[m4] base_pos_w={base_pos[0].cpu().numpy()}  cam_pos_w={cam_pos[0].cpu().numpy()}")
        print(f"[m4] intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  img={W}x{H}")
        print(
            f"[m4] rgb mean={rgb.mean().item():.1f} (non-blank if >0); depth valid frac="
            f"{np.mean((depth > 0) & np.isfinite(depth)):.2f}"
        )

        objects = [n for n in args_cli.objects]
        worst_frame_err = 0.0
        n_checked = 0
        for obj in objects:
            obj_pos = unwrapped.scene[obj].data.root_pos_w[0:1]  # (1,3) world

            # Ground truth in base frame (same T_world_base used for pose_mat -> base offset cancels).
            p_base_gt = subtract_frame_transforms(base_pos, base_quat, obj_pos, identity)[0][0].cpu().numpy()

            # Object center in the camera's OpenCV frame -> pixel + analytic z-depth.
            p_cam = subtract_frame_transforms(cam_pos, cam_quat, obj_pos, identity)[0][0].cpu().numpy()
            Xc, Yc, Zc = p_cam
            if Zc <= 0:
                print(f"[m4]   {obj}: behind camera (Z={Zc:.3f}); skip")
                continue
            u = fx * Xc / Zc + cx
            v = fy * Yc / Zc + cy
            if not (0 <= u < W and 0 <= v < H):
                print(f"[m4]   {obj}: projects out of frame (u={u:.0f},v={v:.0f}); skip")
                continue

            # FRAME GATE: back-project with analytic Z via GaP's formula -> isolates pose_mat correctness.
            Xb, Yb = (u - cx) * Zc / fx, (v - cy) * Zc / fy
            p_bp = pose_mat_np @ np.array([Xb, Yb, Zc, 1.0])
            frame_err = float(np.linalg.norm(p_bp[:3] - p_base_gt))

            # DEPTH SANITY: back-project the RENDERED depth at that pixel; XY should match GT, Z is surface.
            Zr = float(depth[int(round(v)), int(round(u))])
            Xr, Yr = (u - cx) * Zr / fx, (v - cy) * Zr / fy
            p_surf = pose_mat_np @ np.array([Xr, Yr, Zr, 1.0])
            xy_err = float(np.linalg.norm(p_surf[:2] - p_base_gt[:2]))

            worst_frame_err = max(worst_frame_err, frame_err)
            n_checked += 1
            print(
                f"[m4]   {obj}: px=({u:.0f},{v:.0f}) frame_err={frame_err * 100:.3f}cm | "
                f"depth Zr={Zr:.3f} vs Zc={Zc:.3f} xy_err={xy_err * 100:.2f}cm"
            )

        passed = n_checked > 0 and worst_frame_err < _TOL_M
        print(
            f"[m4] RESULT frame-check {'PASS' if passed else 'FAIL'} - "
            f"worst frame_err={worst_frame_err * 100:.3f}cm over {n_checked} objects (tol={_TOL_M * 100:.1f}cm)"
        )
        env.close()


if __name__ == "__main__":
    main()
