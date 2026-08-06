# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run and record one deterministic Newton Gear Assembly rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gear",
        choices=("gear_small", "gear_medium", "gear_large"),
        required=True,
        help="Gear to pick up and insert.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for rl-video-step-0.mp4 and summary.json.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from gymnasium.wrappers import RecordVideo

    import isaaclab.utils.math as math_utils

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.specs import NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE, NEWTON_GEAR_OFFSETS
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    class RenderDecimationWrapper(gym.Wrapper):
        """Reuse a viewport frame for four control steps while recording."""

        def __init__(self, env):
            super().__init__(env)
            self._frame = None
            self._render_count = 0

        def render(self):
            if self._frame is None or self._render_count % 4 == 0:
                self._frame = self.env.render()
            self._render_count += 1
            return self._frame

    action_scale = 0.5
    arena_env = GearAssemblyEnvironment().build(
        GearAssemblyEnvironmentCfg(
            embodiment="droid_differential_ik",
            physics_backend="newton",
            enable_cameras=False,
        )
    )
    arena_env.name = f"gear_assembly_newton_rollout_{args.gear}_{args.seed}"
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, seed=args.seed))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.events.randomize_gear_type.params["gear_types"] = [args.gear]
    env_cfg.terminations.success.params["consecutive_success_steps"] = 100_000
    env_cfg.viewer.resolution = (1280, 720)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = builder.make_registered(env_cfg, env_kwargs, render_mode="rgb_array")
    env = RenderDecimationWrapper(env)
    env = RecordVideo(
        env,
        video_folder=str(args.output_dir),
        step_trigger=lambda step: step == 0,
        video_length=2400,
        disable_logger=True,
    )
    reapply_viewer_cfg(env)

    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    gear = uenv.scene[f"factory_{args.gear}"]
    base = uenv.scene["factory_gear_base"]
    ee_index = robot.body_names.index("base_link")
    desired_ee_quat = robot.data.body_quat_w.torch[:, ee_index].clone()
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=uenv.device)
    total_steps = 0
    phase_steps: dict[str, int] = {}
    max_joint_speed = 0.0
    max_gear_speed = 0.0

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal max_gear_speed, max_joint_speed, total_steps
        _, _, terminated, truncated, _ = env.step(action)
        total_steps += 1
        assert torch.isfinite(robot.data.joint_pos.torch).all(), "Robot joint position became non-finite"
        assert torch.isfinite(robot.data.joint_vel.torch).all(), "Robot joint velocity became non-finite"
        assert torch.isfinite(gear.data.root_link_pose_w.torch).all(), "Gear pose became non-finite"
        assert torch.isfinite(gear.data.root_link_vel_w.torch).all(), "Gear velocity became non-finite"
        max_joint_speed = max(max_joint_speed, torch.abs(robot.data.joint_vel.torch).max().item())
        max_gear_speed = max(
            max_gear_speed,
            torch.linalg.norm(gear.data.root_link_vel_w.torch[:, :3], dim=-1).max().item(),
        )
        return terminated, truncated

    def hold(name: str, count: int, close: bool) -> None:
        print(f"[rollout] {name}: {count} steps", flush=True)
        action.zero_()
        action[:, -1] = float(close)
        for _ in range(count):
            step()
        phase_steps[name] = count

    def move_gear(name: str, target: torch.Tensor, count: int, max_step: float, tolerance: float) -> float:
        print(f"[rollout] {name}: at most {count} steps", flush=True)
        action.zero_()
        action[:, -1] = 1.0
        grasp_offset = gear.data.root_link_pos_w.torch - robot.data.body_pos_w.torch[:, ee_index]
        final_error = float("inf")
        for phase_step in range(count):
            if phase_step and phase_step % 100 == 0:
                print(f"[rollout] {name}: {phase_step}/{count} steps", flush=True)
            current_offset = gear.data.root_link_pos_w.torch - robot.data.body_pos_w.torch[:, ee_index]
            assert torch.linalg.norm(current_offset - grasp_offset).item() <= 0.02, f"Grasp lost during {name}"

            error = target - gear.data.root_link_pos_w.torch
            final_error = torch.linalg.norm(error, dim=-1).item()
            if final_error <= tolerance:
                phase_steps[name] = phase_step
                return final_error

            norm = torch.linalg.norm(error, dim=-1, keepdim=True)
            action[:, :3] = error * torch.clamp(max_step / torch.clamp(norm, min=1.0e-9), max=1.0) / action_scale

            ee_pos = robot.data.body_pos_w.torch[:, ee_index]
            ee_quat = robot.data.body_quat_w.torch[:, ee_index]
            _, orientation_error = math_utils.compute_pose_error(
                ee_pos,
                ee_quat,
                ee_pos,
                desired_ee_quat,
                rot_error_type="axis_angle",
            )
            norm = torch.linalg.norm(orientation_error, dim=-1, keepdim=True)
            action[:, 3:6] = (
                orientation_error * torch.clamp(0.02 / torch.clamp(norm, min=1.0e-9), max=1.0) / action_scale
            )
            step()

        phase_steps[name] = count
        return final_error

    def move_ee(name: str, target: torch.Tensor, count: int, max_step: float, tolerance: float) -> float:
        print(f"[rollout] {name}: at most {count} steps", flush=True)
        action.zero_()
        final_error = float("inf")
        for phase_step in range(count):
            error = target - robot.data.body_pos_w.torch[:, ee_index]
            final_error = torch.linalg.norm(error, dim=-1).item()
            if final_error <= tolerance:
                phase_steps[name] = phase_step
                return final_error
            norm = torch.linalg.norm(error, dim=-1, keepdim=True)
            action[:, :3] = error * torch.clamp(max_step / torch.clamp(norm, min=1.0e-9), max=1.0) / action_scale
            step()
        phase_steps[name] = count
        return final_error

    try:
        hold("open_settle", 20, close=False)
        pregrasp_position = gear.data.root_link_pos_w.torch.clone()
        hold("close", 75, close=True)

        lift_target = gear.data.root_link_pos_w.torch.clone()
        lift_target[:, 2] += 0.10
        lift_error = move_gear("lift", lift_target, 180, 0.005, 0.002)
        lifted_height = (gear.data.root_link_pos_w.torch[:, 2] - pregrasp_position[:, 2]).item()
        hold("post_lift_settle", 30, close=True)

        base_target = base.data.root_link_pos_w.torch + math_utils.quat_apply(
            base.data.root_link_quat_w.torch,
            torch.tensor(NEWTON_GEAR_OFFSETS[args.gear], device=uenv.device).expand(uenv.num_envs, -1),
        )
        base_target[:, 2] = base.data.root_link_pos_w.torch[:, 2] + NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE[args.gear]
        above_target = base_target.clone()
        above_target[:, 2] += 0.10
        large_gear = args.gear == "gear_large"
        transport_error = move_gear(
            "transport",
            above_target,
            1400 if large_gear else 1000,
            0.003 if large_gear else 0.005,
            0.00025,
        )

        insertion_target = base_target.clone()
        insertion_target[:, 2] -= 0.005
        move_gear("insert", insertion_target, 260, 0.003, 0.0015)
        insertion_error = torch.linalg.norm(base_target - gear.data.root_link_pos_w.torch, dim=-1).item()

        retreat_target = robot.data.body_pos_w.torch[:, ee_index].clone()
        retreat_target[:, 2] += 0.08
        hold("release", 75, close=False)
        released_position = gear.data.root_link_pos_w.torch.clone()
        retreat_error = move_ee("retreat", retreat_target, 120, 0.005, 0.003)
        hold("post_release_settle", 30, close=False)

        success_cfg = uenv.termination_manager.get_term_cfg("success")
        success_cfg.params["consecutive_success_steps"] = 10
        uenv.termination_manager.set_term_cfg("success", success_cfg)
        uenv.termination_manager.reset()
        action.zero_()
        success = False
        print("[rollout] success_settle: at most 120 steps", flush=True)
        for success_step in range(1, 121):
            terminated, _ = step()
            if terminated.item():
                success = True
                break
        phase_steps["success_settle"] = success_step

        summary = {
            "gear": args.gear,
            "seed": args.seed,
            "success": success,
            "total_steps": total_steps,
            "phase_steps": phase_steps,
            "base_target_position": base_target[0].tolist(),
            "final_gear_position": released_position[0].tolist(),
            "lifted_height_m": lifted_height,
            "lift_error_m": lift_error,
            "transport_error_m": transport_error,
            "insertion_error_m": insertion_error,
            "retreat_error_m": retreat_error,
            "max_joint_speed_rad_s": max_joint_speed,
            "max_gear_speed_m_s": max_gear_speed,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("ROLLOUT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
        assert success, f"Gear Assembly rollout failed: {summary}"
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    with SimulationAppContext(args):
        _run(args)


if __name__ == "__main__":
    main()
