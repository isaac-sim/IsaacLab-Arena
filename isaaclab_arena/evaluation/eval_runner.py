# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback
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


def load_env(arena_env_args: list[str], job_name: str):

    reload_arena_modules()

    args_parser = get_isaaclab_arena_environments_cli_parser()

    arena_env_args_cli = args_parser.parse_args(arena_env_args)
    arena_builder = get_arena_builder_from_cli(arena_env_args_cli)

    env_name, env_cfg = arena_builder.build_registered()

    # Set unique dataset filename for this job to avoid file locking conflicts
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders.dataset_filename = f"dataset_{job_name}"

    env = arena_builder.make_registered(env_cfg)
    # Don't reset here - rollout_policy() will reset the env. Every reset triggers a new episode, initializing recorder & creating a new hdf5 entry.
    return env


def get_policy_from_job(job: Job) -> "PolicyBase":
    """
    Create a policy from a job configuration. Two paths are supported:
    1. JSON → dict → ConfigDataclass → init cls (preferred, if policy has config_class)
    2. JSON → dict → CLI args → init cls (if policy has add_args_to_parser() and from_args())
    """
    # Each job can be evaluated with a different policy checkpoint, or even a different policy type
    policy_cls = get_policy_cls(job.policy_type)

    # Use direct from_dict if the policy class has config_class defined
    if hasattr(policy_cls, "config_class") and policy_cls.config_class is not None:
        # Use the inherited from_dict() method from PolicyBase
        policy = policy_cls.from_dict(job.policy_config_dict)
    else:
        policy_args_parser = get_isaaclab_arena_cli_parser()
        policy_added_args_parser = policy_cls.add_args_to_parser(policy_args_parser)
        policy_args = policy_added_args_parser.parse_args(job.policy_config_dict)
        policy = policy_cls.from_args(policy_args)
    return policy


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()
    # TODO(xinjieyao, 2026-01-08): support multiple environments simulation as each job may need different number of environments
    assert args_cli.num_envs == 1, "Evaluation runner only supports single environment simulation"

    with SimulationAppContext(args_cli):
        add_eval_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        assert os.path.exists(
            args_cli.eval_jobs_config
        ), f"eval_jobs_config file does not exist: {args_cli.eval_jobs_config}"

        with open(args_cli.eval_jobs_config) as f:
            eval_jobs_config = json.load(f)
        job_manager = JobManager(eval_jobs_config["jobs"])
        metrics_logger = MetricsLogger()

        job_manager.print_jobs_info()

        for job in job_manager:
            if job is not None:
                env = None
                try:
                    # Modules reloading first, otherwise 2 instances of same class are created (e.g. Enum)
                    env = load_env(job.arena_env_args, job.name)

                    policy = get_policy_from_job(job)

                    # priority of setting num_steps is: job -> policy -> args_cli
                    # jobs may need different num_steps than the default num_steps from the policy or the args_cli
                    if job.num_steps is None:
                        if policy.has_length():
                            job.num_steps = policy.length()
                        else:
                            job.num_steps = args_cli.num_steps

                    metrics = rollout_policy(env, policy, num_steps=job.num_steps)

                    job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)

                    # users may not specify metrics for a task, although it's not recommended
                    if metrics is not None:
                        metrics_logger.append_job_metrics(job.name, metrics)

                except Exception as e:
                    # continue with the next job even if one fails
                    job_manager.complete_job(job, metrics={}, status=Status.FAILED)
                    print(f"Job {job.name} failed with error: {e}")
                    print(f"Traceback: {traceback.format_exc()}")

                finally:
                    # Only stop env if it was successfully created
                    if env is not None:
                        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
                        # cleanup managers, including recorder manager closing hdf5 file
                        env.close()

        job_manager.print_jobs_info()
        metrics_logger.print_metrics()

        # Exit with non-zero code if any jobs failed
        job_counts = job_manager.get_job_count()
        if job_counts[Status.FAILED.value] > 0:
            failed_count = job_counts[Status.FAILED.value]
            total_count = len(job_manager.all_jobs)
            raise RuntimeError(f"{failed_count}/{total_count} jobs failed. See logs above for details.")


if __name__ == "__main__":
    main()
