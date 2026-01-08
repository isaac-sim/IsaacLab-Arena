# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import tqdm
import json

import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher
simulation_app = AppLauncher()


def load_env(job):
    from isaaclab_arena.utils.reload_modules import reload_arena_modules

    reload_arena_modules()
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )
    args_parser = get_isaaclab_arena_environments_cli_parser()

    args_cli = args_parser.parse_args(job.args_cli)
    arena_builder = get_arena_builder_from_cli(args_cli)
    env = arena_builder.make_registered()
    env.reset()
    return env

def rollout_policy(env):
    NUM_STEPS = 100
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

def stop_env(env):
    from isaaclab.sim import SimulationContext

    simulation_context = SimulationContext.instance()
    simulation_context._disable_app_control_on_stop_handle = True
    simulation_context.stop()
    simulation_context.clear_instance()
    env.close()

    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    omni.usd.get_context().new_stage()

def main():
    from isaaclab_arena.evaluation.eval_jobs import JobManager, Status
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.metrics.metrics import compute_metrics
    from isaaclab_arena.metrics.metrics_logger import MetricsLogger
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args()
    with open("isaaclab_arena/evaluation/eval_jobs_config.json", "r") as f:
        eval_jobs_config = json.load(f)
    job_manager = JobManager(eval_jobs_config["jobs"])

    metrics_logger = MetricsLogger()

    print(f"Starting job processing. Job counts: {job_manager.get_job_count()}")

    while not job_manager.is_empty():
        job = job_manager.get_next_job()
        if job is not None:
            try:
                env = load_env(job)
                rollout_policy(env)
                metrics = compute_metrics(env)
                job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)
                stop_env(env)
                metrics_logger.append_job_metrics(job.name, metrics)
            except Exception as e:
                job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                print(f"Job {job.name} failed with error: {e}")

    print(f"All jobs processed. Final job counts: {job_manager.get_job_count()}")
    metrics_logger.print_metrics()
if __name__ == "__main__":
    main()
