# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""M4 bridge client: stream the REAL exterior camera (rgb + depth + intrinsics + pose_mat) to GaP.

Extends the M3 motion client with real perception data so GaP's perceive subgraph (DINO/SAM3/VLM->OBB)
runs on actual ILA renders. The exterior camera is aimed agentview-style at runtime (good for open-vocab
recognition); pose_mat is read from the camera's actual pose, so the R5 convention (verified at 0.000cm)
holds regardless of where the camera is aimed.

pose_mat = T_world_base^-1 @ T_world_cam(quat_w_ros), OpenCV optical, depth = distance_to_image_plane.
Sent under camera key robot0_robotview (no "eye_in_hand" substring -> GaP treats it as the exterior view).

Run order:
  1) (gap venv, terminal A) set -a; source ~/.config/gap/vlm.env; set +a && MUJOCO_GL=egl uv run gap run
     <graph> --real franka --no-rr-autostart --no-video
  2) (ILA venv, terminal B)
     ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m4_bridge_client.py \
         --headless --num_envs 1 --enable_cameras --placement_seed 0 libero_object_packing --control joint_pos
"""

from __future__ import annotations

import argparse
import socket
import struct
import time

import msgpack
import msgpack_numpy
import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

msgpack_numpy.patch()

_CAM = "exterior_cam"
_DEPTH_DT = "distance_to_image_plane"
_GRIPPER_OPEN_M = 0.04
_GRIPPER_THRESH = 0.5
_SETTLE_STEPS = 25
# Agentview-style external view: from +X, elevated, looking back/down at the table center.
_CAM_EYE = (1.3, 0.0, 0.65)
_CAM_TARGET = (0.32, 0.0, 0.08)


def add_bridge_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("GaP Bridge Arguments")
    group.add_argument("--bridge_host", type=str, default="127.0.0.1")
    group.add_argument("--bridge_port", type=int, default=9000)
    group.add_argument("--bridge_camera_name", type=str, default="robot0_robotview")


def _send(sock: socket.socket, obj: dict) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return buf


def _recv(sock: socket.socket) -> dict:
    (n,) = struct.unpack("!I", _recvn(sock, 4))
    return msgpack.unpackb(_recvn(sock, n), raw=False)


def _connect_with_retry(host: str, port: int, timeout_s: float = 300.0, interval_s: float = 1.0) -> socket.socket:
    print(f"[m4] env ready; connecting to GaP server {host}:{port} (retrying up to {timeout_s:.0f}s)")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.create_connection((host, port))
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(interval_s)


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    add_bridge_args(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import torch
        from isaaclab.utils.math import (
            create_rotation_matrix_from_view,
            matrix_from_quat,
            subtract_frame_transforms,
        )

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
        robot = unwrapped.scene["robot"]
        cam = unwrapped.scene[_CAM]
        has_fingers = robot.data.joint_pos.shape[1] > 7

        # Hold home (absolute joint-pos control); without this a zero action would drive joints to 0.
        action = torch.zeros(env.action_space.shape, device=device)
        action[..., :7] = robot.data.joint_pos[0, :7]
        if action.shape[-1] > 7:
            action[..., 7] = 1.0  # gripper open

        # Settle, then aim the exterior camera agentview-style and render from there.
        for _ in range(_SETTLE_STEPS):
            env.step(action)
        eye = torch.tensor([_CAM_EYE], device=device)
        tgt = torch.tensor([_CAM_TARGET], device=device)
        cam.set_world_poses_from_view(eye, tgt)
        for _ in range(8):
            env.step(action)

        # Deterministic pose_mat = T_base_cam (OpenCV), consistent with the render BY CONSTRUCTION:
        # create_rotation_matrix_from_view returns the OpenGL orientation set_world_poses_from_view uses;
        # OpenCV = R_gl @ diag(1,-1,-1). Avoids the stale/ambiguous quat_w_ros buffer after aiming.
        R_gl = create_rotation_matrix_from_view(eye, tgt, "Z", device=device)[0]
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        T_wc = torch.eye(4, device=device)
        T_wc[:3, :3] = R_gl @ flip
        T_wc[:3, 3] = eye[0]
        base_pos, base_quat = robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1]
        T_wb = torch.eye(4, device=device)
        T_wb[:3, :3] = matrix_from_quat(base_quat)[0]
        T_wb[:3, 3] = base_pos[0]
        pose_mat = torch.linalg.inv(T_wb) @ T_wc
        pose_mat_np = pose_mat.cpu().numpy().astype(np.float32)
        K_np = cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)
        print(f"[m4] exterior cam aimed eye={_CAM_EYE} -> pose_mat t={pose_mat_np[:3,3]}")

        # Ground-truth object centers in the BASE frame (the frame GaP's OBB will be in) for gate scoring.
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        for obj in args_cli.objects:
            op = unwrapped.scene[obj].data.root_pos_w[0:1]
            p_base = subtract_frame_transforms(base_pos, base_quat, op, identity)[0][0].cpu().numpy()
            print(f"[m4] GT_BASE {obj} = [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}]")

        def camera_frame() -> dict:
            rgb = np.ascontiguousarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            depth = cam.data.output[_DEPTH_DT][0].squeeze(-1).cpu().numpy().astype(np.float32)
            depth = np.ascontiguousarray(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0))
            # Depth goes under the TOP-LEVEL "depth_data" key: that is what GaP's _convert_observation
            # actually reads (franka_real_env.py:387), despite the contract doc showing images.depth.
            return {
                "images": {"rgb": rgb},
                "depth_data": depth,
                "intrinsics": {"left": {"intrinsics_matrix": K_np}},
                "pose_mat": pose_mat_np,
            }

        tick = 0
        with _connect_with_retry(args_cli.bridge_host, args_cli.bridge_port) as sock:
            print(f"[m4] connected to GaP server {args_cli.bridge_host}:{args_cli.bridge_port}")
            while True:
                q = robot.data.joint_pos[0, :7].cpu().numpy()
                finger = float(robot.data.joint_pos[0, 7].cpu()) if has_fingers else 0.0
                gripper_frac = float(np.clip(finger / _GRIPPER_OPEN_M, 0.0, 1.0))
                obs = {
                    "timestamp": time.time(),
                    "left": {"joint_pos": [float(v) for v in q] + [gripper_frac]},
                    args_cli.bridge_camera_name: camera_frame(),
                }
                _send(sock, obs)

                try:
                    incoming = _recv(sock)
                except ConnectionError:
                    print(f"[m4] server closed the connection after {tick} ticks; exiting")
                    break

                left = incoming.get("left") or {}
                cmd = left.get("joint_pos")
                if cmd is not None:
                    action[..., :7] = torch.tensor(np.asarray(cmd, dtype=np.float32).reshape(-1)[:7], device=device)
                gripper = left.get("gripper")
                if gripper is not None and action.shape[-1] > 7:
                    action[..., 7] = 1.0 if float(gripper) >= _GRIPPER_THRESH else -1.0

                env.step(action)
                tick += 1
                if tick % 50 == 0:
                    print(f"[m4] tick={tick} streaming rgb+depth+pose_mat under '{args_cli.bridge_camera_name}'")

        env.close()


if __name__ == "__main__":
    main()
