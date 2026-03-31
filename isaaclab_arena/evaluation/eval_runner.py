# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import json
import os
import traceback
from gymnasium.wrappers import RecordVideo
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.eval_runner_cli import add_eval_runner_arguments
from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status
from isaaclab_arena.evaluation.policy_runner import get_policy_cls, rollout_policy
from isaaclab_arena.metrics.metrics_logger import MetricsLogger
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.utils.reload_modules import reload_arena_modules
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase


def load_env(arena_env_args: list[str], job_name: str, render_mode: str | None = None):

    reload_arena_modules()

    args_parser = get_isaaclab_arena_environments_cli_parser()

    arena_env_args_cli = args_parser.parse_args(arena_env_args)
    arena_builder = get_arena_builder_from_cli(arena_env_args_cli)

    env_name, env_cfg = arena_builder.build_registered()

    # Set unique dataset filename for this job to avoid file locking conflicts
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job_name}"

    env = arena_builder.make_registered(env_cfg, render_mode=render_mode)
    # Don't reset here - rollout_policy() will reset the env. Every reset triggers a new episode, initializing recorder & creating a new hdf5 entry.
    return env


def enable_cameras_if_required(eval_jobs_config: dict, args_cli: argparse.Namespace) -> None:
    """
    Check if any job requires cameras and enable them in args_cli if needed. Users can set
    enable_cameras: true in individual job config, or add --enable_cameras to the CLI.
    Camera support must be enabled when the simulation starts, not during individual job execution.

    Args:
        eval_jobs_config: Dictionary containing job configurations
        args_cli: CLI arguments namespace to modify
    """
    for job_dict in eval_jobs_config["jobs"]:
        if "arena_env_args" in job_dict and job_dict["arena_env_args"].get("enable_cameras", False):
            if not hasattr(args_cli, "enable_cameras") or not args_cli.enable_cameras:
                args_cli.enable_cameras = True
            break


def get_policy_from_job(job: Job) -> "PolicyBase":
    """
    Create a policy from a job configuration. Two paths are supported:
    1. JSON → dict → ConfigDataclass → init cls (preferred, if policy has config_class)
    2. JSON → dict → CLI args → init cls (if policy has add_args_to_parser() and from_args())
    """
    # Each job can be evaluated with a different policy checkpoint, or even a different policy type
    policy_cls = get_policy_cls(job.policy_type)

    policy_config_dict = dict(job.policy_config_dict)
    # Align policy num_envs with env when the policy config supports it (optional key)
    if hasattr(policy_cls, "config_class") and policy_cls.config_class is not None:
        config_fields = {f.name for f in dataclasses.fields(policy_cls.config_class)}
        if "num_envs" in config_fields:
            policy_config_dict["num_envs"] = job.num_envs

    # Use direct from_dict if the policy class has config_class defined
    if hasattr(policy_cls, "config_class") and policy_cls.config_class is not None:
        # Use the inherited from_dict() method from PolicyBase
        policy = policy_cls.from_dict(policy_config_dict)
    else:
        policy_args_parser = get_isaaclab_arena_cli_parser()
        policy_added_args_parser = policy_cls.add_args_to_parser(policy_args_parser)
        policy_args = policy_added_args_parser.parse_args(policy_config_dict)
        policy = policy_cls.from_args(policy_args)
    return policy


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()

    # Load job configuration before starting simulation to check requirements
    add_eval_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    assert not args_cli.distributed, "Distributed evaluation is not supported yet"

    assert os.path.exists(
        args_cli.eval_jobs_config
    ), f"eval_jobs_config file does not exist: {args_cli.eval_jobs_config}"

    with open(args_cli.eval_jobs_config, encoding="utf-8") as f:
        eval_jobs_config = json.load(f)

    # Check if any job requires cameras and enable them if needed before starting simulation
    enable_cameras_if_required(eval_jobs_config, args_cli)

    with SimulationAppContext(args_cli):
        job_manager = JobManager(eval_jobs_config["jobs"])
        metrics_logger = MetricsLogger()

        job_manager.print_jobs_info()

        if args_cli.video:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            print(f"[INFO] Video recording enabled. Videos will be saved to: {args_cli.video_dir}")

        for job in job_manager:
            if job is not None:
                env = None
                try:
                    render_mode = "rgb_array" if args_cli.video else None
                    env = load_env(job.arena_env_args, job.name, render_mode=render_mode)

                    policy = get_policy_from_job(job)

                    # Resolve simulation length: num_steps and num_episodes are mutually exclusive.
                    # Priority: job config -> policy length -> CLI default
                    if job.num_steps is None and job.num_episodes is None:
                        if policy.has_length():
                            job.num_steps = policy.length()
                        else:
                            job.num_steps = args_cli.num_steps

                    if args_cli.video:
                        if job.num_steps is not None:
                            video_length = job.num_steps
                        else:
                            video_length = job.num_episodes * env.unwrapped.max_episode_length
                        video_kwargs = {
                            "video_folder": os.path.join(args_cli.video_dir, job.name),
                            "step_trigger": lambda step: step == 0,
                            "video_length": video_length,
                            "disable_logger": True,
                        }
                        print(f"[INFO] Recording video for job '{job.name}' -> {video_kwargs['video_folder']}")
                        env = RecordVideo(env, **video_kwargs)

                    metrics = rollout_policy(
                        env,
                        policy,
                        num_steps=job.num_steps,
                        num_episodes=job.num_episodes,
                        language_instruction=job.language_instruction,
                    )

                    job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)

                    # users may not specify metrics for a task, although it's not recommended
                    if metrics is not None:
                        metrics_logger.append_job_metrics(job.name, metrics)

                except Exception as e:
                    job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                    print(f"Job {job.name} failed with error: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    if not args_cli.continue_on_error:
                        raise

                finally:
                    # Only stop env if it was successfully created
                    if env is not None:
                        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
                        # cleanup managers, including recorder manager closing hdf5 file
                        env.close()

        job_manager.print_jobs_info()
        metrics_logger.print_metrics()


if __name__ == "__main__":
    main()
