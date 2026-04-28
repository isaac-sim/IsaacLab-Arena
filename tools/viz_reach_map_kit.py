"""Visualize a Franka EEF reachability map inside the Isaac Lab Kit viewer.

Either loads a precomputed reach map (``--load_npz tools/franka_reach_top_down.npz``)
or computes one on the fly with batched cuRobo IK in the robot base frame,
then scatters Isaac Lab sphere markers at feasible voxels in world frame so
you can orbit the viewer and see the workspace overlaid on the actual scene.

Run inside the curobo container::

  docker exec isaaclab_arena-curobo bash -c \\
    'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh tools/viz_reach_map_kit.py \\
       --viz kit --num_envs 1 avocadoPnPbowltable --embodiment franka_ik \\
       --load_npz tools/franka_reach_top_down.npz'
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)


def add_args(parser) -> None:
    parser.add_argument("--load_npz", type=str, default="",
                        help="Path to precomputed NPZ; skip cuRobo recompute.")
    parser.add_argument("--grid", type=int, default=22, help="voxels per axis (recompute mode)")
    parser.add_argument("--x_min", type=float, default=-0.4)
    parser.add_argument("--x_max", type=float, default=1.0)
    parser.add_argument("--y_min", type=float, default=-0.9)
    parser.add_argument("--y_max", type=float, default=0.9)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=1.4)
    parser.add_argument("--dwell_steps", type=int, default=4000)


def _compute_reach_map_with_curobo(env, args_cli):
    """Run batched cuRobo IK; return (positions_base [M,3], pos_err [M]) on env device."""
    from curobo.types.math import Pose
    from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
    from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

    planner_cfg = CuroboPlannerCfg.franka_config()
    planner_cfg.visualize_plan = False
    planner_cfg.visualize_spheres = False
    planner = CuroboPlanner(
        env=env.unwrapped, robot=env.unwrapped.scene["robot"],
        config=planner_cfg, env_id=0,
    )
    planner.update_world()
    ik = planner.motion_gen.ik_solver
    device = planner.tensor_args.device

    N = int(args_cli.grid)
    xs = torch.linspace(args_cli.x_min, args_cli.x_max, N, device=device)
    ys = torch.linspace(args_cli.y_min, args_cli.y_max, N, device=device)
    zs = torch.linspace(args_cli.z_min, args_cli.z_max, N, device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    positions_base = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    quat_top_down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    quaternions = quat_top_down.expand(positions_base.shape[0], 4).contiguous()
    targets = Pose(position=positions_base, quaternion=quaternions)

    print(f"[reach_map] solving IK for {positions_base.shape[0]} voxels (grid={N}^3) ...",
          flush=True)
    result = ik.solve_batch(targets)
    success = result.success.view(-1).bool()
    pos_err = result.position_error.view(-1)
    return positions_base[success], pos_err[success]


def _load_reach_map_from_npz(path: str, device: torch.device):
    """Load NPZ produced by tools/compute_reach_map.py. Returns (positions_base [M,3], pos_err [M])."""
    d = np.load(path)
    success = d["success"]                         # (Nx, Ny, Nz)
    pos_err = d["pos_err"]                         # (Nx, Ny, Nz)
    xs, ys, zs = d["x"], d["y"], d["z"]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    feas = success.astype(bool)
    pts_base = np.stack([X[feas], Y[feas], Z[feas]], axis=-1)
    errs = pos_err[feas]
    print(f"[reach_map] loaded {pts_base.shape[0]} feasible voxels from {path}", flush=True)
    return (
        torch.from_numpy(pts_base).float().to(device),
        torch.from_numpy(errs).float().to(device),
    )


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        parser2 = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser2.parse_args()

        builder = get_arena_builder_from_cli(args_cli)
        env, _ = builder.make_registered_and_return_cfg()
        env.reset()

        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(zero)
        device = env.unwrapped.device

        # ---- get feasible voxel positions (in robot base frame)
        if args_cli.load_npz:
            feas_pos_b, feas_err = _load_reach_map_from_npz(args_cli.load_npz, device)
        else:
            feas_pos_b, feas_err = _compute_reach_map_with_curobo(env, args_cli)

        if feas_pos_b.shape[0] == 0:
            print("[reach_map] no feasible voxels — nothing to visualize", flush=True)
            env.close()
            return 0

        median_err = float(feas_err.median().item())
        bin_idx = (feas_err > median_err).long()       # 0=tight, 1=loose

        # ---- transform feasible voxels into world frame
        from isaaclab.utils.math import quat_apply

        import warp as wp

        robot = env.unwrapped.scene["robot"]
        base_pos = wp.to_torch(robot.data.root_pos_w)[0, :3].to(device)
        base_quat = wp.to_torch(robot.data.root_quat_w)[0, :4].to(device)
        feas_pos_w = (
            quat_apply(
                base_quat.unsqueeze(0).expand(feas_pos_b.shape[0], 4),
                feas_pos_b,
            )
            + base_pos
        )

        # ---- reach-map markers
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        from isaaclab.sim.spawners.shapes import SphereCfg

        reach_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/reach_map",
            markers={
                "tight": SphereCfg(
                    radius=0.012,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
                ),
                "loose": SphereCfg(
                    radius=0.012,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.95)),
                ),
            },
        )
        VisualizationMarkers(reach_cfg).visualize(
            translations=feas_pos_w, marker_indices=bin_idx,
        )
        print(
            f"[reach_map] spawned {feas_pos_w.shape[0]} reach markers — "
            f"green=tight (pos_err<median), cyan=loose. Median pos_err = "
            f"{median_err * 1000:.2f} mm",
            flush=True,
        )

        # ---- robot base marker (large red sphere at origin of robot base frame)
        base_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/reach_map_base",
            markers={
                "base": SphereCfg(
                    radius=0.06,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                ),
            },
        )
        VisualizationMarkers(base_cfg).visualize(translations=base_pos.unsqueeze(0))
        print(
            f"[reach_map] base marker at world pos "
            f"({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})",
            flush=True,
        )

        # ---- dwell so the viewer stays open
        for _ in range(int(args_cli.dwell_steps)):
            env.step(zero)
        env.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
