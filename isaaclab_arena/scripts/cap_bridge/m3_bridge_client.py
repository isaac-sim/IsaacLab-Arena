# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""M3 bridge client: drive the libero_object_packing env over GaP's real-robot msgpack protocol.

This is the ILA side of the GaP<->ILA bridge (Isaac-cap docs/ila_side_design.md M3). It mirrors the
reference stub (Isaac-cap src/isaac_cap/stub_ila_client.py) framing verbatim, but replaces the toy
joint integrator with the real env: each tick it sends an observation, receives GaP's absolute
joint+gripper target, applies it via the env's absolute joint-position action term, and steps the sim.

GaP hosts the server; this connects as the client. GaP reset() blocks until the first RGB frame, so we
send an observation before reading the first action. For the motion smoke the camera is a blank frame
(perception is M4) published under --bridge_camera_name (default robot0_robotview, GaP's default).

Run order:
  1) (gap venv, terminal A) cd graph-as-policy && uv run gap run <graph> --real franka --no-rr-autostart --no-video
  2) (ILA venv, terminal B)
     ./dev_run.sh isaaclab_arena/scripts/cap_bridge/m3_bridge_client.py \
         --headless --num_envs 1 libero_object_packing --control joint_pos
"""

from __future__ import annotations

import argparse
import numpy as np
import socket
import struct
import time

import msgpack
import msgpack_numpy

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

msgpack_numpy.patch()

# Panda finger joint position (m) at fully open; used to normalize the gripper fraction 0..1.
_GRIPPER_OPEN_M = 0.04
# GaP's continuous gripper fraction is mapped to the binary action term at this threshold.
_GRIPPER_THRESH = 0.5


def add_bridge_args(parser: argparse.ArgumentParser) -> None:
    """Bridge connection flags. Registered top-level (before the env subparser)."""
    group = parser.add_argument_group("GaP Bridge Arguments")
    group.add_argument("--bridge_host", type=str, default="127.0.0.1", help="GaP msgpack server host.")
    group.add_argument("--bridge_port", type=int, default=9000, help="GaP msgpack server port.")
    group.add_argument(
        "--bridge_camera_name",
        type=str,
        default="robot0_robotview",
        help="Observation camera key (must match GaP's configured camera_names).",
    )
    group.add_argument(
        "--bridge_max_ticks",
        type=int,
        default=0,
        help="Stop after N exchanges (0 = run until the server closes the connection).",
    )


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
    """Connect to the GaP server, retrying while it is not yet listening.

    The env takes minutes to boot, but GaP reset() only waits ~60s for the first RGB frame. Booting the
    env first and retrying the connect lets the operator start GaP after the env is ready, so GaP's clock
    starts when we are about to stream -- not during our boot.
    """
    print(f"[m3] env ready; connecting to GaP server {host}:{port} (retrying up to {timeout_s:.0f}s)")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.create_connection((host, port))
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(interval_s)


def _blank_camera(height: int = 64, width: int = 64) -> dict:
    """Blank RGB-D + plausible intrinsics/identity pose: enough to unblock GaP reset(), not perception."""
    intrinsics = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float32)
    return {
        "images": {
            "rgb": np.full((height, width, 3), 128, dtype=np.uint8),
            "depth": np.ones((height, width), dtype=np.float32),
        },
        "intrinsics": {"left": {"intrinsics_matrix": intrinsics}},
        "pose_mat": np.eye(4, dtype=np.float32),
    }


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    add_bridge_args(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import torch

        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli = args_parser.parse_args()

        env = get_arena_builder_from_cli(args_cli).make_registered()
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        has_fingers = robot.data.joint_pos.shape[1] > 7

        # Start from a hold-open action; absent gripper commands then keep the gripper open.
        action = torch.zeros(env.action_space.shape, device=device)
        if action.shape[-1] > 7:
            action[..., 7] = 1.0

        target_q = None
        tick = 0
        with _connect_with_retry(args_cli.bridge_host, args_cli.bridge_port) as sock:
            print(f"[m3] connected to GaP server {args_cli.bridge_host}:{args_cli.bridge_port}")
            while True:
                q = robot.data.joint_pos[0, :7].cpu().numpy()
                finger = float(robot.data.joint_pos[0, 7].cpu()) if has_fingers else 0.0
                gripper_frac = float(np.clip(finger / _GRIPPER_OPEN_M, 0.0, 1.0))

                obs = {
                    "timestamp": time.time(),
                    "left": {"joint_pos": [float(v) for v in q] + [gripper_frac]},
                    args_cli.bridge_camera_name: _blank_camera(),
                }
                _send(sock, obs)

                try:
                    incoming = _recv(sock)
                except ConnectionError:
                    print(f"[m3] server closed the connection after {tick} ticks; exiting")
                    break

                left = incoming.get("left") or {}
                cmd = left.get("joint_pos")
                if cmd is not None:
                    target_q = np.asarray(cmd, dtype=np.float32).reshape(-1)[:7]
                    action[..., :7] = torch.from_numpy(target_q).to(device)
                gripper = left.get("gripper")
                if gripper is not None and action.shape[-1] > 7:
                    action[..., 7] = 1.0 if float(gripper) >= _GRIPPER_THRESH else -1.0

                env.step(action)

                tick += 1
                if tick % 50 == 0:
                    err = float(np.linalg.norm(q - target_q)) if target_q is not None else float("nan")
                    print(f"[m3] tick={tick} joint_err={err:.4f} gripper_frac={gripper_frac:.2f}")
                if args_cli.bridge_max_ticks and tick >= args_cli.bridge_max_ticks:
                    print(f"[m3] reached --bridge_max_ticks={args_cli.bridge_max_ticks}; exiting")
                    break

        env.close()


if __name__ == "__main__":
    main()
