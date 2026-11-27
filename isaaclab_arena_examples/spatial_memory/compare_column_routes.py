# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare storing routes and storing free passages on identical G1 head-camera depth surveys."""

from __future__ import annotations

import argparse
import itertools
import json
import numpy as np
from pathlib import Path


def route_lengths_m(routes_y_E, requests_y_E):
    """Measure each route for each pair of inspection points and receiving dock."""
    from isaaclab_arena_environments.spatial_memory.tasks.column_grid import crossing_distance_m

    return np.asarray([[crossing_distance_m(route, request) for request in requests_y_E] for route in routes_y_E])


def select_route_library(lengths_m, size, margin_m):
    """Choose routes before reveal to cover the most requests, breaking ties by mean excess distance."""
    shortest_m = lengths_m.min(axis=0)

    def rank(indices):
        excess_m = lengths_m[list(indices)].min(axis=0) - shortest_m
        return (int((excess_m > margin_m).sum()), float(excess_m.sum()))

    return list(min(itertools.combinations(range(len(lengths_m)), size), key=rank))


def score_survey(estimated_open_y_E, true_open_y_E, requests_y_E):
    """Score matched route choices against physical openings and the requested distance budget."""
    from isaaclab_arena_environments.spatial_memory.tasks.column_grid import DISTANCE_MARGIN_M, OPENING_TOLERANCE_M

    routes = np.asarray(list(itertools.product(*estimated_open_y_E)))
    true_routes = list(itertools.product(*true_open_y_E))
    lengths = route_lengths_m(routes, requests_y_E)
    budget = route_lengths_m(true_routes, requests_y_E).min(axis=0) + DISTANCE_MARGIN_M
    fits = (np.abs(routes[:, :, None] - true_open_y_E).min(axis=2) < OPENING_TOLERANCE_M).all(axis=1)
    result = {}
    for label, size in (("one_route", 1), ("two_routes", 2), ("four_routes", 4), ("map", len(routes))):
        library = select_route_library(lengths, size, DISTANCE_MARGIN_M)
        selected = np.asarray(library)[lengths[library].argmin(axis=0)]
        result[label] = int((fits[selected] & (lengths[selected, np.arange(len(budget))] <= budget)).sum())
    # Two complementary routes contain every surveyed opening. Switch between them at the gantries.
    left_route, right_route = np.sort(estimated_open_y_E, axis=1).T
    spliced_routes = np.asarray(list(itertools.product(*zip(left_route, right_route, strict=True))))
    spliced_lengths = route_lengths_m(spliced_routes, requests_y_E)
    selected = spliced_lengths.argmin(axis=0)
    passages = spliced_routes[selected]
    fits = (np.abs(passages[:, :, None] - true_open_y_E).min(axis=2) < OPENING_TOLERANCE_M).all(axis=1)
    result["two_spliceable_routes"] = int((fits & (spliced_lengths.min(axis=0) <= budget)).sum())
    return result


def count_column_returns(camera, env_origins):
    """Count calibrated depth returns inside each hypothesized column footprint."""
    import torch

    from isaaclab.utils.math import transform_points, unproject_depth

    from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
    from isaaclab_arena_environments.spatial_memory.tasks.column_grid import LANE_Y_E, ROW_X_E

    stride = 4
    depth = camera.data.output["distance_to_image_plane"].torch[:, ::stride, ::stride]
    intrinsics = camera.data.intrinsic_matrices.torch.clone()
    intrinsics[:, :2] /= stride
    points_C = unproject_depth(depth, intrinsics)
    t_E_C = camera.data.pos_w.torch - env_origins
    points_E = transform_points(points_C, t_E_C, camera.data.quat_w_ros.torch)
    above_floor = (points_E[:, :, 2] > G1_FLOOR_Z_M + 0.08) & (points_E[:, :, 2] < 0.05)
    counts = torch.zeros((len(depth), len(ROW_X_E), len(LANE_Y_E)), device=depth.device)
    for row, x_E in enumerate(ROW_X_E):
        in_row = above_floor & ((points_E[:, :, 0] - x_E).abs() < 0.19)
        for lane, y_E in enumerate(LANE_Y_E):
            in_lane = (points_E[:, :, 1] - y_E).abs() < 0.40
            counts[:, row, lane] = (in_row & in_lane).sum(dim=1)
    return counts


def survey_layouts(env, num_layouts, pose_seed):
    """Survey fixed inspection poses with jitter and retain the estimated and actual openings for scoring."""
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.tasks.column_grid import LANE_Y_E, ROW_X_E

    base = env.unwrapped
    camera = base.scene["robot_head_cam"]
    robot = base.scene["robot"]
    rng = np.random.default_rng(pose_seed)
    estimates, layouts = [], []
    while len(layouts) < num_layouts:
        env.reset()
        counts = torch.zeros(base.num_envs, len(ROW_X_E), len(LANE_Y_E), device=base.device)
        for x_E in ROW_X_E:
            xy_E = np.asarray([x_E + 1.55, -2.80]) + rng.uniform(-0.02, 0.02, (base.num_envs, 2))
            yaw_jitter = rng.uniform(-0.015, 0.015, base.num_envs)
            for yaw in (2.54, 1.90):
                T_W_pelvis = robot.data.root_pose_w.torch.clone()
                T_W_pelvis[:, :2] = base.scene.env_origins[:, :2] + T_W_pelvis.new_tensor(xy_E)
                angle = T_W_pelvis.new_tensor(yaw + yaw_jitter)
                T_W_pelvis[:, 3:] = 0
                T_W_pelvis[:, 5] = torch.sin(angle / 2)
                T_W_pelvis[:, 6] = torch.cos(angle / 2)
                robot.write_root_pose_to_sim_index(root_pose=T_W_pelvis)
                for _ in range(6):
                    base.sim.step(render=True)
                    base.scene.update(base.physics_dt)
                camera.update(base.step_dt, force_recompute=True)
                counts += count_column_returns(camera, base.scene.env_origins)
        lanes = counts.argsort(dim=2, stable=True)[:, :, :2].sort(dim=2).values.cpu().numpy()
        estimates.extend(np.asarray(LANE_Y_E)[lanes])
        layouts.extend(get_episode_state(env).layout_params["open_y_E"].cpu().numpy().copy())
    return np.asarray(estimates[:num_layouts]), np.asarray(layouts[:num_layouts])


def main():
    """Render surveys and write paired results per layout."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=96)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--layout_seed", type=int, default=817217)
    parser.add_argument("--pose_seed", type=int, default=963214)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.layouts < 2 or args.num_envs < 1:
        parser.error("At least two layouts and one environment are required.")
    args.enable_cameras = True
    with SimulationAppContext(args):
        from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
        from isaaclab_arena_environments.spatial_memory.tasks.column_grid import LANE_Y_E

        env = make_spatial_memory_env(
            "spatial_memory_column_grid", args.num_envs, seed=args.layout_seed, enable_cameras=True, with_depth=True
        )
        try:
            estimates, truth = survey_layouts(env, args.layouts, args.pose_seed)
        finally:
            env.close()
        requests = list(itertools.product(LANE_Y_E, repeat=3))
        results = []
        for i, actual in enumerate(truth):
            counts = score_survey(estimates[i], actual, requests)
            counts["swapped_map"] = score_survey(estimates[(i + 1) % len(truth)], actual, requests)["map"]
            results.append(counts)
        report = {
            "layout_seed": args.layout_seed,
            "pose_seed": args.pose_seed,
            "num_envs": args.num_envs,
            "queries_per_layout": len(requests),
            "valid_choices_per_layout": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
