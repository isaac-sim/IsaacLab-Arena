# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one RoboLab exact task and verify that rigid references reset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = REPO_ROOT / "isaaclab_arena_environments" / "robolab_exact" / "tasks"


def main() -> None:
    """Build, perturb, and reset one exact environment."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, help="Exact task YAML basename.")
    known_args, remaining_args = parser.parse_known_args()
    arena_parser = get_isaaclab_arena_cli_parser()
    args = arena_parser.parse_args(["--num_envs", "1", *remaining_args])

    with SimulationAppContext(args):
        import torch

        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.evaluation.resource_cleanup import close_environment

        task_path = TASKS_DIR / known_args.task
        spec = ArenaEnvGraphSpec.from_yaml(task_path)
        builder = ArenaEnvBuilder(spec.to_arena_env(), arena_env_builder_cfg_from_argparse(args))
        manager_cfg, gym_kwargs = builder.compose_manager_cfg()
        manager_cfg.recorders = {}
        manager_cfg.episode_recorders = {}
        env = builder.make_registered(manager_cfg, gym_kwargs)

        try:
            env.reset()
            reference_names = [reference.id for reference in spec.object_references or []]
            initial_poses = {name: env.unwrapped.scene[name].data.root_pose_w.torch.clone() for name in reference_names}
            for name in reference_names:
                asset = env.unwrapped.scene[name]
                perturbed_pose = initial_poses[name].clone()
                perturbed_pose[:, 0] += 1.0
                asset.write_root_pose_to_sim(perturbed_pose)

            env.reset()
            for name, initial_pose in initial_poses.items():
                actual_pose = env.unwrapped.scene[name].data.root_pose_w.torch
                torch.testing.assert_close(actual_pose, initial_pose, atol=1e-5, rtol=0.0)
            print(f"[runtime] {task_path.name}: reset {len(reference_names)} rigid references")
        finally:
            close_environment(env)


if __name__ == "__main__":
    main()
