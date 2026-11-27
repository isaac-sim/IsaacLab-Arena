# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fit a supervised lane-choice policy on depth clouds from prescribed poses or actual G1 inspection walks."""

import argparse
import json
import math
import numpy as np
import tempfile
from pathlib import Path


def surface_histograms(cloud):
    """Count valid measured surfaces in a fixed volume behind the common pickup pose."""
    rows = []
    for points_R in cloud.cpu().numpy():
        xyz_R = points_R[points_R[:, 3] > 0, :3]
        counts, _ = np.histogramdd(xyz_R, bins=(9, 32, 12), range=((-2.6, -0.65), (-2.25, 2.25), (-0.65, 0.9)))
        rows.append(np.log1p(counts).flatten())
    return np.asarray(rows)


def lane_labels(base, task):
    """Read supervision and the delayed query separately from the policy input."""
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M

    episode = get_episode_state(base)
    params = episode.layout_params
    if task in ("partition_delivery", "partition_silhouettes"):
        centers = []
        for name in episode.logic.obstacle_names():
            marker_y_E = base.arena_world.get_pose_e(name)[:, 1]
            centers.append(marker_y_E + LANE_PITCH_M / 2 - 0.075)
        y_E = torch.stack(centers, dim=1)
        queries = params["color"]
    else:
        y_E = params["t_E_target_y"][:, None]
        queries = torch.zeros_like(y_E[:, 0], dtype=torch.long)
    return (y_E / LANE_PITCH_M + 1.5).round().long().cpu().numpy(), queries.cpu().numpy()


def survey_batch(env, rng, survey="endpoints"):
    """Return accumulated and current-only observations after the same survey and turn to pickup."""
    import torch

    from isaaclab_arena_environments.spatial_memory.observations import get_online_map_provider

    base = env.unwrapped
    provider = get_online_map_provider(base)
    robot = base.scene["robot"]
    camera = base.scene["robot_head_cam"]
    ids = torch.arange(base.num_envs, device=base.device)
    provider.reset(base, ids)
    xy_E = rng.uniform(-0.025, 0.025, (base.num_envs, 2))
    yaw_offset = rng.uniform(-0.02, 0.02, base.num_envs)
    headings = (math.pi - 0.75, -math.pi + 0.75)
    if survey == "sweep":
        headings = np.linspace(math.pi - 0.75, math.pi + 0.75, 11)
        joints = robot.data.joint_pos.torch.clone()
        joint_velocity = torch.zeros_like(joints)
        root_velocity = joints.new_zeros((base.num_envs, 6))
    for yaw in (*headings, 0.0):
        T_W_R = robot.data.root_pose_w.torch.clone()
        T_W_R[:, :3] = base.scene.env_origins
        T_W_R[:, :2] += T_W_R.new_tensor(xy_E)
        angle = T_W_R.new_tensor(yaw + yaw_offset)
        T_W_R[:, 3:] = 0
        T_W_R[:, 5] = torch.sin(angle / 2)
        T_W_R[:, 6] = torch.cos(angle / 2)
        robot.write_root_pose_to_sim_index(root_pose=T_W_R)
        for _ in range(6):
            if survey == "sweep":
                robot.write_root_pose_to_sim_index(root_pose=T_W_R)
                robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
                robot.write_joint_position_to_sim_index(position=joints)
                robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
            base.sim.step(render=True)
            base.scene.update(base.physics_dt)
        camera.update(base.step_dt, force_recompute=True)
        cloud = provider.query(base)
    accumulated = surface_histograms(cloud)
    provider.reset(base, ids)
    return accumulated, surface_histograms(provider.query(base))


def inspection_controller(base, logic):
    """Start the common inspection and pickup motion in each environment."""
    controller = logic.embodiment.make_scripted_controller(base)
    for i in range(base.num_envs):
        controller.set_plan(i, logic.inspection_and_pickup())
    return controller


def walking_survey(env, obs):
    """Capture both clouds when each walking robot first receives its public instruction."""
    import torch

    from isaaclab_arena_environments.spatial_memory.depth_cloud import DepthCloudProvider
    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.observations import get_online_map_provider

    base = env.unwrapped
    episode = get_episode_state(base)
    logic = episode.logic
    controller = inspection_controller(base, logic)
    accumulated = get_online_map_provider(base)
    current = DepthCloudProvider(base, accumulated.camera_name, accumulated.far_m, accumulated.blocked_frames)
    pending = torch.ones(base.num_envs, device=base.device, dtype=torch.bool)
    clouds, frames = [None] * base.num_envs, [None] * base.num_envs
    for _ in range(base.max_episode_length):
        ready = (pending & obs["task_obs"]["instruction"].any(dim=1)).nonzero().flatten()
        if len(ready):
            current.reset(base, ready)
            for i, cloud, frame in zip(
                ready.tolist(),
                surface_histograms(obs["mapped"]["online_map"][ready]),
                surface_histograms(current.query(base)[ready]),
                strict=True,
            ):
                clouds[i], frames[i] = cloud, frame
            pending[ready] = False
            if not pending.any():
                return clouds, frames
        obs, _, terminated, truncated, _ = env.step(controller.action(base, logic, episode))
        if (pending & (terminated | truncated)).any():
            raise RuntimeError(
                "An inspection ended before the instruction. Include this acquisition failure in results."
            )
    raise RuntimeError("Some walking surveys timed out before the instruction.")


def fit_lane_weights(features, labels):
    """Fit ridge classifiers for the four lane choices for each possible request."""
    train = np.column_stack((features, np.ones(len(features))))
    classes = np.eye(4)[labels].reshape(len(train), -1)
    return train.T @ np.linalg.solve(train @ train.T + np.eye(len(train)), classes)


def select_lanes(weights, features, queries):
    """Choose a lane from each cloud and its public request."""
    inputs = np.column_stack((features, np.ones(len(features))))
    scores = (inputs @ weights).reshape(len(inputs), -1, 4)
    return scores[np.arange(len(inputs)), queries].argmax(axis=1)


def deliver_one_batch(env, weights):
    """Run the learned lane decision with the G1 walking and manipulation controller on fresh layouts."""
    import torch

    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LANE_PITCH_M

    base = env.unwrapped
    obs, _ = env.reset()
    episode = get_episode_state(base)
    logic = episode.logic
    controller = inspection_controller(base, logic)
    chosen = [None] * base.num_envs
    ended = set()
    with tempfile.TemporaryDirectory() as folder, torch.inference_mode():
        path = Path(folder) / "deliveries.jsonl"
        base.episode_recorder.set_output_path(path)
        for _ in range(base.max_episode_length):
            request = obs["task_obs"]["instruction"]
            for i in request.any(dim=1).nonzero().flatten().tolist():
                if chosen[i] is None and i not in ended:
                    feature = surface_histograms(obs["mapped"]["online_map"][i : i + 1])
                    chosen[i] = int(select_lanes(weights, feature, [int(request[i].argmax())])[0])
                    lane_y_E = (chosen[i] - 1.5) * LANE_PITCH_M
                    controller.plans[i].waypoints.extend(logic.deliver_to_lane(lane_y_E))
            obs, _, done, timeout, _ = env.step(controller.action(base, logic, episode))
            ended.update((done | timeout).nonzero().flatten().tolist())
            if len(ended) == base.num_envs:
                break
        first = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            first.setdefault(record["env_id"], record)
        assert len(first) == base.num_envs, "Every assessed layout must finish."
    return [{"chosen_lane": chosen[i], "episode": first[i]} for i in range(base.num_envs)]


def main():
    """Evaluate lane choices on disjoint layouts and optionally execute fresh deliveries."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("reverse_gap", "shelf_bay", "partition_delivery", "partition_silhouettes"), required=True
    )
    parser.add_argument("--train_layouts", type=int, default=128)
    parser.add_argument("--test_layouts", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=720941)
    parser.add_argument("--survey", choices=("endpoints", "sweep", "walking"), default="endpoints")
    parser.add_argument("--deliver", action="store_true", help="Also execute one new batch of physical deliveries.")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.train_layouts < 1 or args.test_layouts < 2 or args.num_envs < 1:
        parser.error("Use at least one training layout, two held-out layouts and one environment.")
    args.enable_cameras = True
    with SimulationAppContext(args):
        import torch

        from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env

        env = make_spatial_memory_env(
            f"spatial_memory_{args.task}", args.num_envs, args.seed, observation_variant="mapped", enable_cameras=True
        )
        total = args.train_layouts + args.test_layouts
        clouds, frames, labels, queries = [], [], [], []
        try:
            with torch.inference_mode():
                for batch in range(math.ceil(total / args.num_envs)):
                    obs, _ = env.reset()
                    lanes, request = lane_labels(env.unwrapped, args.task)
                    if args.survey == "walking":
                        accumulated, frame = walking_survey(env, obs)
                    else:
                        accumulated, frame = survey_batch(env, np.random.default_rng(55121 + batch), args.survey)
                    clouds.extend(accumulated)
                    frames.extend(frame)
                    labels.extend(lanes)
                    queries.extend(request)
            labels = np.asarray(labels[:total])
            queries = np.asarray(queries[:total])
            test = np.arange(args.train_layouts, total)
            report = {
                "task": args.task,
                "seed": args.seed,
                "survey": args.survey,
                "train_layouts": args.train_layouts,
                "test_layouts": args.test_layouts,
                "query": queries[test].tolist(),
                "correct_lane": labels[test, queries[test]].tolist(),
            }
            for key, feature in (("accumulated", clouds), ("current", frames)):
                values = np.asarray(feature[:total])
                weights = fit_lane_weights(values[: args.train_layouts], labels[: args.train_layouts])
                report[key] = select_lanes(weights, values[test], queries[test]).tolist()
                if key == "accumulated":
                    report["swapped"] = select_lanes(weights, np.roll(values[test], 1, axis=0), queries[test]).tolist()
                    cloud_weights = weights
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if args.deliver:
                with torch.inference_mode():
                    report["deliveries"] = deliver_one_batch(env, cloud_weights)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.close()


if __name__ == "__main__":
    main()
