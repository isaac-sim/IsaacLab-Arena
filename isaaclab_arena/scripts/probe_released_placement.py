# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure whether the ETL released-placement predicate is attainable under physics.

This is a calibration probe, not a demonstration generator. It initializes a deterministic batch
of sugar-box poses above the bowl, writes zero object velocity once, and then advances physics
without touching the object again.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    """One object initialization relative to the destination root."""

    name: str
    expected_positive: bool
    offset_x_m: float
    offset_y_m: float
    initial_vertical_offset_m: float
    yaw_rad: float


def make_candidates() -> list[Candidate]:
    """Return the preregistered positive sweep followed by geometric controls."""

    candidates = []
    xy_offsets = ((0.0, 0.0), (-0.01, 0.0), (0.01, 0.0), (0.0, -0.01), (0.0, 0.01))
    for vertical_offset in (0.16, 0.20, 0.24):
        for yaw in (0.0, math.pi / 4.0, math.pi / 2.0):
            for offset_x, offset_y in xy_offsets:
                candidates.append(
                    Candidate(
                        name=(
                            f"positive_z{vertical_offset:.2f}_yaw{yaw:.3f}_"
                            f"xy{offset_x:+.2f}_{offset_y:+.2f}"
                        ),
                        expected_positive=True,
                        offset_x_m=offset_x,
                        offset_y_m=offset_y,
                        initial_vertical_offset_m=vertical_offset,
                        yaw_rad=yaw,
                    )
                )
    for index, (offset_x, offset_y) in enumerate(((0.04, 0.0), (-0.04, 0.0), (0.0, 0.04))):
        candidates.append(
            Candidate(
                name=f"negative_horizontal_{index}",
                expected_positive=False,
                offset_x_m=offset_x,
                offset_y_m=offset_y,
                initial_vertical_offset_m=0.20,
                yaw_rad=0.0,
            )
        )
    return candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-control-steps", type=int, default=45)
    parser.add_argument("--observe-control-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assert args.settle_control_steps > 0
    assert args.observe_control_steps > 0

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app

    import torch
    import warp as wp

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.predicates.released_placement import released_placement_condition
    from isaaclab_arena.utils.physics_settle import step_physics
    from isaaclab_arena_environments.etl_pnp_maple_table_environment import (
        EtlPnpMapleTableEnvironment,
        EtlPnpMapleTableEnvironmentCfg,
    )

    candidates = make_candidates()
    task_cfg = EtlPnpMapleTableEnvironmentCfg(
        embodiment="droid_abs_joint_pos",
        enable_cameras=False,
        pick_up_object="sugar_box_ycb_robolab",
        destination_location="bowl_ycb_robolab",
    )
    arena_env = EtlPnpMapleTableEnvironment().build(task_cfg)
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=1,
        seed=args.seed,
        placement_seed=args.seed,
        resolve_on_reset=False,
    )
    env = ArenaEnvBuilder(arena_env, builder_cfg).make_registered()
    env.reset()

    try:
        unwrapped = env.unwrapped
        scene = unwrapped.scene
        device = torch.device(unwrapped.device)
        object_entity = scene[task_cfg.pick_up_object]
        destination_entity = scene[task_cfg.destination_location]
        robot = scene["robot"]
        contact_sensor = scene[arena_env.task.contact_sensor_name]
        decimation = int(unwrapped.cfg.decimation)
        joint_names = list(robot.data.joint_names)
        body_names = list(robot.data.body_names)
        gripper_joint_index = joint_names.index("finger_joint")
        end_effector_body_index = body_names.index("base_link")

        def read_state() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            pose = wp.to_torch(object_entity.data.root_link_pose_w)
            velocity = wp.to_torch(object_entity.data.root_link_vel_w)
            destination = wp.to_torch(destination_entity.data.root_link_pose_w)[:, :3]
            gripper = wp.to_torch(robot.data.joint_pos)[:, gripper_joint_index]
            end_effector = wp.to_torch(robot.data.body_pos_w)[:, end_effector_body_index, :]
            force = torch.linalg.vector_norm(wp.to_torch(contact_sensor.data.force_matrix_w), dim=-1).reshape(-1)

            condition = released_placement_condition(
                object_pose_w=pose,
                object_velocity_w=velocity,
                destination_position_w=destination,
                gripper_joint_position=gripper,
                end_effector_position_w=end_effector,
                destination_contact_force=force,
                max_horizontal_offset=task_cfg.placement_max_horizontal_offset_m,
                min_vertical_offset=task_cfg.placement_vertical_offset_range_m[0],
                max_vertical_offset=task_cfg.placement_vertical_offset_range_m[1],
                max_axis_tilt=task_cfg.placement_max_axis_tilt_rad,
                max_linear_speed=task_cfg.placement_max_linear_speed_m_s,
                max_angular_speed=task_cfg.placement_max_angular_speed_rad_s,
                max_open_joint_position=task_cfg.placement_max_open_joint_position_rad,
                min_end_effector_distance=task_cfg.placement_min_end_effector_distance_m,
                min_contact_force=task_cfg.placement_min_contact_force_n,
            )

            relative_position = pose[:, :3] - destination
            quaternion = pose[:, 3:] / torch.linalg.vector_norm(pose[:, 3:], dim=-1, keepdim=True).clamp_min(
                torch.finfo(pose.dtype).eps
            )
            vertical_alignment = torch.abs(1.0 - 2.0 * (quaternion[:, 0].square() + quaternion[:, 1].square()))
            metrics = {
                "horizontal_offset_m": torch.linalg.vector_norm(relative_position[:, :2], dim=-1),
                "vertical_offset_m": relative_position[:, 2],
                "axis_tilt_rad": torch.arccos(vertical_alignment.clamp(-1.0, 1.0)),
                "linear_speed_m_s": torch.linalg.vector_norm(velocity[:, :3], dim=-1),
                "angular_speed_rad_s": torch.linalg.vector_norm(velocity[:, 3:], dim=-1),
                "gripper_joint_position_rad": gripper,
                "end_effector_distance_m": torch.linalg.vector_norm(end_effector - pose[:, :3], dim=-1),
                "contact_force_n": force,
                "object_pose_w_xyzw": pose,
            }
            return condition, metrics

        rows = []
        env_ids = torch.zeros(1, dtype=torch.long, device=device)
        zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
        for trial_index, candidate in enumerate(candidates):
            env.reset()
            destination_position = wp.to_torch(destination_entity.data.root_link_pose_w)[:, :3]
            object_pose = torch.zeros((1, 7), dtype=torch.float32, device=device)
            object_pose[0, :3] = destination_position[0]
            object_pose[0, 0] += candidate.offset_x_m
            object_pose[0, 1] += candidate.offset_y_m
            object_pose[0, 2] += candidate.initial_vertical_offset_m
            object_pose[0, 5] = math.sin(candidate.yaw_rad / 2.0)
            object_pose[0, 6] = math.cos(candidate.yaw_rad / 2.0)
            object_entity.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
            object_entity.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

            for _ in range(args.settle_control_steps):
                step_physics(env, decimation)

            consecutive = torch.zeros(1, dtype=torch.long, device=device)
            maximum_consecutive = torch.zeros_like(consecutive)
            final_metrics = {}
            for _ in range(args.observe_control_steps):
                step_physics(env, decimation)
                condition, final_metrics = read_state()
                consecutive = torch.where(condition, consecutive + 1, torch.zeros_like(consecutive))
                maximum_consecutive = torch.maximum(maximum_consecutive, consecutive)

            metrics = {
                key: value[0].detach().cpu().tolist() if value.ndim > 1 else float(value[0].item())
                for key, value in final_metrics.items()
            }
            passed = bool(maximum_consecutive[0].item() >= task_cfg.placement_dwell_steps)
            rows.append(
                {
                    "candidate": asdict(candidate),
                    "max_consecutive_valid_control_steps": int(maximum_consecutive[0].item()),
                    "passed": passed,
                    "final": metrics,
                }
            )
            print(
                f"[{trial_index + 1:02d}/{len(candidates)}] {candidate.name}: passed={passed}, "
                f"dwell={int(maximum_consecutive[0].item())}, "
                f"xy={metrics['horizontal_offset_m']:.4f}, z={metrics['vertical_offset_m']:.4f}, "
                f"tilt={metrics['axis_tilt_rad']:.3f}, force={metrics['contact_force_n']:.3f}"
            )

        positive_rows = [row for row in rows if row["candidate"]["expected_positive"]]
        negative_rows = [row for row in rows if not row["candidate"]["expected_positive"]]
        positive_pass_count = sum(row["passed"] for row in positive_rows)
        negative_pass_count = sum(row["passed"] for row in negative_rows)
        result = {
            "experiment": "EXP-016-released-placement-feasibility",
            "predicate_version": task_cfg.success_predicate_version,
            "seed": args.seed,
            "num_envs": 1,
            "num_trials": len(candidates),
            "physics_dt_s": float(unwrapped.sim.get_physics_dt()),
            "decimation": decimation,
            "control_dt_s": float(unwrapped.step_dt),
            "settle_control_steps": args.settle_control_steps,
            "observe_control_steps": args.observe_control_steps,
            "dwell_steps": task_cfg.placement_dwell_steps,
            "thresholds": {
                "max_horizontal_offset_m": task_cfg.placement_max_horizontal_offset_m,
                "vertical_offset_range_m": list(task_cfg.placement_vertical_offset_range_m),
                "max_axis_tilt_rad": task_cfg.placement_max_axis_tilt_rad,
                "max_linear_speed_m_s": task_cfg.placement_max_linear_speed_m_s,
                "max_angular_speed_rad_s": task_cfg.placement_max_angular_speed_rad_s,
                "max_open_joint_position_rad": task_cfg.placement_max_open_joint_position_rad,
                "min_end_effector_distance_m": task_cfg.placement_min_end_effector_distance_m,
                "min_contact_force_n": task_cfg.placement_min_contact_force_n,
            },
            "summary": {
                "positive_candidates": len(positive_rows),
                "positive_pass_count": positive_pass_count,
                "negative_controls": len(negative_rows),
                "negative_pass_count": negative_pass_count,
                "gate_passed": positive_pass_count > 0 and negative_pass_count == 0,
            },
            "trials": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
