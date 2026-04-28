"""Compute Franka EEF reachability map with batched cuRobo IK on GPU.

Pure cuRobo + PyTorch — no Isaac Sim / Kit boot. Runs in seconds for a
20-30 voxel cube. Saves a .npz with positions + IK outcome and an
optional 3-D matplotlib voxel render.

Run inside the curobo container:

  docker exec isaaclab_arena-curobo bash -c \\
    'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh tools/compute_reach_map.py \\
       --grid 25 --save_npz tools/franka_reach_top_down.npz \\
       --save_png tools/franka_reach_top_down.png'
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=25, help="voxels per axis")
    p.add_argument("--x_min", type=float, default=-0.4)
    p.add_argument("--x_max", type=float, default=1.0)
    p.add_argument("--y_min", type=float, default=-0.9)
    p.add_argument("--y_max", type=float, default=0.9)
    p.add_argument("--z_min", type=float, default=0.0)
    p.add_argument("--z_max", type=float, default=1.4)
    p.add_argument("--num_seeds", type=int, default=20, help="cuRobo IK seeds per query")
    p.add_argument("--robot_yml", type=str, default="franka.yml")
    p.add_argument(
        "--quat_wxyz",
        type=float,
        nargs=4,
        default=[0.0, 1.0, 0.0, 0.0],
        help="EE orientation, wxyz. Default = top-down (rotate 180° about x).",
    )
    p.add_argument("--save_npz", type=str, default="")
    p.add_argument("--save_png", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ---- cuRobo IK solver ---------------------------------------------------
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tdt = TensorDeviceType()
    print(f"[reach_map] device = {tdt.device}", flush=True)

    robot_dict = load_yaml(join_path(get_robot_configs_path(), args.robot_yml))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_dict, tdt)
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, num_seeds=int(args.num_seeds), tensor_args=tdt
    )
    ik = IKSolver(ik_cfg)

    # ---- 3-D grid in robot base frame --------------------------------------
    N = int(args.grid)
    x = torch.linspace(args.x_min, args.x_max, N, device=tdt.device)
    y = torch.linspace(args.y_min, args.y_max, N, device=tdt.device)
    z = torch.linspace(args.z_min, args.z_max, N, device=tdt.device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    positions = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    quat = torch.tensor(args.quat_wxyz, device=tdt.device, dtype=positions.dtype)
    quaternions = quat.expand(positions.shape[0], 4).contiguous()
    targets = Pose(position=positions, quaternion=quaternions)

    # ---- batched IK on GPU --------------------------------------------------
    print(f"[reach_map] solving IK for {positions.shape[0]} voxels (grid={N}^3)…", flush=True)
    t0 = time.time()
    result = ik.solve_batch(targets)
    torch.cuda.synchronize() if tdt.device.type == "cuda" else None
    dt = time.time() - t0

    success = result.success.view(N, N, N).cpu().numpy()
    pos_err = result.position_error.view(N, N, N).cpu().numpy()
    rot_err = result.rotation_error.view(N, N, N).cpu().numpy()
    feasible_count = int(success.sum())
    total = int(success.size)
    print(
        f"[reach_map] solved in {dt:.2f}s | feasible voxels: {feasible_count}/{total} "
        f"({100.0 * feasible_count / total:.1f}%)",
        flush=True,
    )

    # ---- save NPZ -----------------------------------------------------------
    if args.save_npz:
        os.makedirs(os.path.dirname(args.save_npz) or ".", exist_ok=True)
        np.savez(
            args.save_npz,
            success=success,
            pos_err=pos_err,
            rot_err=rot_err,
            x=x.cpu().numpy(),
            y=y.cpu().numpy(),
            z=z.cpu().numpy(),
            quat_wxyz=np.array(args.quat_wxyz),
        )
        print(f"[reach_map] saved NPZ -> {args.save_npz}", flush=True)

    # ---- save matplotlib PNG ------------------------------------------------
    if args.save_png and feasible_count > 0:
        import matplotlib

        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt

        # Color voxels by IK position error (lower = greener).
        norm = plt.Normalize(vmin=0.0, vmax=max(float(pos_err[success].max()), 1e-3))
        cmap = plt.cm.viridis
        colors = np.zeros(success.shape + (4,), dtype=float)
        for idx in zip(*np.where(success)):
            colors[idx] = cmap(norm(pos_err[idx]))
        colors[..., 3] *= 0.30  # alpha for see-through (low so base marker is visible)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.voxels(success, facecolors=colors, edgecolor=None)
        ax.set_xlabel("x (base frame)")
        ax.set_ylabel("y (base frame)")
        ax.set_zlabel("z (base frame)")
        ax.set_title(
            f"Franka top-down reachability — {feasible_count}/{total} feasible "
            f"({100*feasible_count/total:.1f}%)"
        )
        # Tick labels in actual base-frame meters, not voxel index.
        ax.set_xticks(np.linspace(0, N, 5))
        ax.set_xticklabels([f"{v:.2f}" for v in np.linspace(args.x_min, args.x_max, 5)])
        ax.set_yticks(np.linspace(0, N, 5))
        ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(args.y_min, args.y_max, 5)])
        ax.set_zticks(np.linspace(0, N, 5))
        ax.set_zticklabels([f"{v:.2f}" for v in np.linspace(args.z_min, args.z_max, 5)])

        # Robot base origin — convert (0,0,0) in base frame to voxel-index space.
        base_xi = N * (0.0 - args.x_min) / (args.x_max - args.x_min)
        base_yi = N * (0.0 - args.y_min) / (args.y_max - args.y_min)
        base_zi = N * (0.0 - args.z_min) / (args.z_max - args.z_min)
        # Vertical dashed plumb line through the base — easy to spot even
        # when the marker itself is occluded by voxels.
        ax.plot(
            [base_xi, base_xi], [base_yi, base_yi], [0, N],
            color="red", linestyle="--", linewidth=1.2, alpha=0.7,
        )
        ax.scatter(
            [base_xi], [base_yi], [base_zi],
            s=300, c="red", marker="o", edgecolor="black", linewidth=2.0,
            label="robot base (0,0,0)", zorder=10,
        )
        # Short axis triad at the base for orientation (red=x, green=y, blue=z).
        triad_len = 0.18 * N
        ax.quiver(base_xi, base_yi, base_zi, triad_len, 0, 0, color="red",   linewidth=2.5)
        ax.quiver(base_xi, base_yi, base_zi, 0, triad_len, 0, color="green", linewidth=2.5)
        ax.quiver(base_xi, base_yi, base_zi, 0, 0, triad_len, color="blue",  linewidth=2.5)
        ax.legend(loc="upper left", fontsize=9)
        # Tilt camera up a bit so the base (at z=0) doesn't sit on the
        # bottom edge of the figure where it's hard to see.
        ax.view_init(elev=22, azim=-55)

        # Colorbar.
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("IK position error (m)")

        os.makedirs(os.path.dirname(args.save_png) or ".", exist_ok=True)
        fig.savefig(args.save_png, dpi=150, bbox_inches="tight")
        print(f"[reach_map] saved PNG -> {args.save_png}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
