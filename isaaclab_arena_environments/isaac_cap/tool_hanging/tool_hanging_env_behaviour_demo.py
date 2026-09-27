# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate that a wrench can settle onto its hook and trigger a success reset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo


def _build_tool_hanging_demo_environment(*, enable_cameras: bool):
    """Build the wrench-easy graph used by the behavior demo."""
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_environments.isaac_cap import register_components

    register_components()
    return ArenaEnvGraphSpec.from_yaml(Path(__file__).with_name("wrench_easy.yaml")).to_arena_env(
        enable_cameras=enable_cameras
    )


class ToolHangingEnvBehaviourDemo(EnvBehaviourDemo):
    """Teleport the wrench over its hook and require a settled success reset."""

    label = "tool-hanging-validation"

    def __init__(self, *args, video_dir: Path | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_dir = video_dir

    def make_env(self):
        """Build the environment and optionally record every observation camera."""
        env = super().make_env()
        if self.video_dir is None:
            return env

        from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

        return wrap_env_for_video(
            env,
            VideoRecordingCfg(
                record_camera_video=True,
                video_base_dir=str(self.video_dir),
                camera_name_prefix="tool-hanging",
            ),
            num_steps=None,
            num_episodes=1,
        )

    def setup_demo(self) -> None:
        """Resolve the wrench, hook, and loop-on-rod goal."""
        import torch

        from isaaclab_arena_environments.isaac_cap.tool_hanging.geometry import LoopOnRod

        assert self.base_env.num_envs == 1, "Tool-hanging validation expects one environment."
        self.torch = torch
        (self.goal,) = self.arena_environment.task.goals
        assert isinstance(self.goal.geometry, LoopOnRod), "Wrench validation requires a loop-on-rod goal."
        self.wrench = self.base_env.scene[self.goal.tool.name]
        self.zero_action = torch.zeros(
            (1, self.base_env.action_manager.total_action_dim),
            device=self.base_env.device,
        )

    def _place_wrench_on_hook(self) -> None:
        """Position the wrench ring around the center of the hook shank."""
        from isaaclab.utils.math import quat_apply, quat_from_matrix

        T_W_H = self.base_env.arena_world.get_pose_w(self.goal.fixture.name)
        rod = self.goal.geometry.rod
        shank_center_H = 0.5 * (T_W_H.new_tensor([rod.start_xyz]) + T_W_H.new_tensor([rod.end_xyz]))
        shank_center_W = T_W_H[:, :3] + quat_apply(T_W_H[:, 3:], shank_center_H)
        # Hang the wrench vertically: local +Z follows the shank and its handle points down.
        R_W_T = T_W_H.new_tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]])
        q_W_T = quat_from_matrix(R_W_T)
        ring_center_T = T_W_H.new_tensor([self.goal.geometry.loops[0].center_xyz])
        T_W_T = self.torch.cat((shank_center_W - quat_apply(q_W_T, ring_center_T), q_W_T), dim=-1)
        self.wrench.write_root_pose_to_sim(T_W_T)
        self.wrench.write_root_velocity_to_sim(self.torch.zeros((1, 6), device=self.base_env.device))

    def run_cycle(self, cycle: int) -> None:
        """Place and settle the wrench, then require the normal success reset."""
        print(f"[{self.label}] cycle {cycle}: place wrench on hook", flush=True)
        self._place_wrench_on_hook()
        with self.torch.inference_mode():
            for step in range(300):
                _, _, terminated, truncated, _ = self.step(self.zero_action)
                success = self.base_env.termination_manager.get_term("success")
                if step == 0:
                    assert not success.any(), "A wrench that has not settled must not count as hung."
                assert not truncated.any(), "Hang validation timed out."
                if terminated.any():
                    assert success.all(), "Episode ended without tool-hanging success."
                    assert self.base_env.episode_length_buf[0] == 0, "Success did not reset the environment."
                    print(f"[{self.label}] cycle {cycle}: success reset observed", flush=True)
                    return
        raise RuntimeError("Wrench did not settle on the hook.")


def run_demo(
    simulation_app,
    *,
    cycles: int = 0,
    real_time: bool = True,
    video_dir: Path | None = None,
) -> None:
    """Run the tool-hanging behavior validation."""
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = ToolHangingEnvBehaviourDemo(
        simulation_app,
        _build_tool_hanging_demo_environment(enable_cameras=video_dir is not None),
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=True, placement_seed=42),
        real_time=real_time,
        video_dir=video_dir,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the tool-hanging validation demo."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument("--no-real-time", action="store_true", help="Run without wall-clock rate limiting.")
    parser.add_argument("--video-dir", type=Path, help="Optional directory for observation-camera recordings.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.limit_cpu_threads = 1
    args.enable_cameras = args.video_dir is not None

    with SimulationAppContext(args) as simulation_app:
        run_demo(
            simulation_app,
            cycles=args.cycles,
            real_time=not args.no_real_time,
            video_dir=args.video_dir,
        )


if __name__ == "__main__":
    main()
