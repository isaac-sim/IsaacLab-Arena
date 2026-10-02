# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate syringe disposal, physical release, settling, and success resets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab_arena_environments.isaac_cap.tools.env_behaviour_demo import EnvBehaviourDemo


class SyringeEnvBehaviourDemo(EnvBehaviourDemo):
    """Place scored syringes in the receiver, then open the gripper to allow success."""

    label = "syringe-validation"

    def __init__(self, *args, video_dir: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_dir = video_dir

    def make_env(self):
        env = super().make_env()
        if self.video_dir is None:
            return env
        from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

        return wrap_env_for_video(
            env,
            VideoRecordingCfg(
                record_camera_video=True, video_base_dir=str(self.video_dir), camera_name_prefix="syringe"
            ),
            num_steps=None,
            num_episodes=1,
        )

    def setup_demo(self):
        import torch

        self.torch = torch
        self.task = self.arena_environment.task
        self.robot = self.base_env.scene["robot"]
        self.arm_indices = [self.robot.joint_names.index(f"fr3_joint{i}") for i in range(1, 8)]
        assert self.base_env.num_envs == 1, "Syringe validation expects one environment"

    def _place_syringe(self, name: str, index: int):
        from isaaclab.utils.math import quat_apply, quat_mul

        syringe = self.base_env.scene[name]
        # W is world, R is the receiver, and S is the syringe root frame.
        T_W_R = self.base_env.arena_world.get_pose_w("sharps_container")
        # Two horizontal columns, with successive pairs settling onto earlier layers.
        # Keep the whole syringe inside the physical container before letting go.
        t_R_S = T_W_R.new_tensor([[0.075 + 0.045 * (index % 2), -0.1225, 0.025 + 0.035 * (index // 2)]])
        T_W_S = T_W_R.clone()
        if len(self.task.objects) <= 2:
            # Release through the aperture with a slight tilt, avoiding an exactly
            # horizontal balancing pose on the syringe's curved collision hulls.
            t_R_S = T_W_R.new_tensor([[0.0975, -0.1225, 0.30]])
            half_angle = math.radians(80) / 2
            q_R_S = T_W_R.new_tensor([[math.sin(half_angle), 0, 0, math.cos(half_angle)]])
            T_W_S[:, 3:] = quat_mul(T_W_R[:, 3:], q_R_S)
        T_W_S[:, :3] += quat_apply(T_W_R[:, 3:], t_R_S)
        syringe.write_root_pose_to_sim_index(root_pose=T_W_S)
        syringe.write_root_velocity_to_sim_index(root_velocity=self.torch.zeros((1, 6), device=self.base_env.device))

    def run_cycle(self, cycle: int):
        """Require disposal to fail while closed and succeed only after opening."""
        action = self.torch.zeros((1, 8), device=self.base_env.device)
        action[:, :7] = self.robot.data.joint_pos.torch[:, self.arm_indices]
        action[:, 7] = 1.0
        for _ in range(75):
            _, _, terminated, truncated, _ = self.step(action)
            assert not (terminated | truncated).any(), "Episode ended before disposal"
        for index, obj in enumerate(self.task.objects):
            self._place_syringe(obj.name, index)
            for _ in range(100):
                _, _, terminated, truncated, _ = self.step(action)
                assert not (terminated | truncated).any(), "Disposal must not succeed with the gripper closed"
        # The blank syringe in designated remains in its tray throughout this cycle.
        action[:, 7] = 0.0
        for step in range(500 * len(self.task.objects)):
            _, _, terminated, truncated, _ = self.step(action)
            success = self.base_env.termination_manager.get_term("success")
            assert not truncated.any(), "Syringe validation timed out"
            if terminated.any():
                assert success.all(), "Episode ended without syringe disposal success"
                assert step >= self.task.consecutive_success_steps - 1, "Success skipped the release dwell"
                assert self.base_env.episode_length_buf[0] == 0, "Success did not reset the environment"
                print(f"[{self.label}] cycle {cycle}: success reset observed", flush=True)
                return
        from isaaclab.utils.math import quat_apply_inverse

        T_W_R = self.base_env.arena_world.get_pose_w("sharps_container")
        for obj in self.task.objects:
            data = self.base_env.scene[obj.name].data
            center_R = quat_apply_inverse(T_W_R[:, 3:], data.root_com_pos_w.torch - T_W_R[:, :3])
            print(
                f"[{self.label}] {obj.name}: COM in receiver={center_R.tolist()}, "
                f"linear velocity={data.root_lin_vel_w.torch.tolist()}, "
                f"angular velocity={data.root_ang_vel_w.torch.tolist()}",
                flush=True,
            )
        print(
            f"[{self.label}] gripper position="
            f"{self.base_env.arena_world.get_joint_position('robot', 'left_driver_joint').tolist()}",
            flush=True,
        )
        raise RuntimeError("Syringes did not settle inside the container after release")


def run_demo(simulation_app, *, variant="single", cycles=0, placement_seed=42, real_time=True, video_dir=None):
    """Run a registered syringe variant through repeatable disposal cycles."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.isaac_cap import register_components

    register_components()
    registry = EnvironmentRegistry()
    factory = registry.get_component_by_name(f"syringe_{variant}_newton")
    cfg = registry.get_environment_cfg_type(factory)(enable_cameras=video_dir is not None)
    demo = SyringeEnvBehaviourDemo(
        simulation_app,
        factory().build(cfg),
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=True, placement_seed=placement_seed),
        real_time=real_time,
        video_dir=video_dir,
    )
    demo.run_demo(cycles)


def main():
    """Launch scripted syringe validation with optional observation videos."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("single", "both", "designated", "cluttered"), default="single")
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--placement-seed", type=int, default=42)
    parser.add_argument("--no-real-time", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = args.video_dir is not None
    args.limit_cpu_threads = 1
    with SimulationAppContext(args) as simulation_app:
        run_demo(
            simulation_app,
            variant=args.variant,
            cycles=args.cycles,
            placement_seed=args.placement_seed,
            real_time=not args.no_real_time,
            video_dir=args.video_dir,
        )


if __name__ == "__main__":
    main()
