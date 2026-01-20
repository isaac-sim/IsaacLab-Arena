# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time
from enum import Enum
from prettytable import PrettyTable
from queue import Queue


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(
        self,
        name: str,
        arena_env_args: dict,
        policy_type: str,
        num_steps: int = None,
        policy_config_dict: dict = None,
        status: Status = None,
    ):
        """Initialize a Job instance.

        Args:
            name: Job name, used to identify the job in the queue and in the logs.
            arena_env_args: Dictionary of arguments for configuring the arena environment
            num_steps: Number of steps to run the policy for
            policy_type: Type of policy to use
            policy_config_dict: Dictionary configuration for the policy.
            status: Job status (defaults to PENDING)
        """
        self.name = name
        self.arena_env_args = arena_env_args
        self.num_steps = num_steps
        self.policy_type = policy_type
        self.policy_config_dict = policy_config_dict if policy_config_dict is not None else {}
        self.status = status if status is not None else Status.PENDING
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create a Job instance from a dictionary.

        Args:
            data: Dictionary containing job data with keys:
                  - name: Job name
                  - arena_env_args: Dictionary of arguments for configuring the arena environment
                  - num_steps: Number of steps to run the policy for
                  - policy_type: Type of policy to use
                  - policy_args: Dictionary of arguments for the policy. These are passed to the policy class's from_args method.
                  - status: Status string (optional, defaults to PENDING)

        Returns:
            New Job instance
        """
        assert "name" in data, "name is required"
        assert "arena_env_args" in data, "arena_env_args is required"
        assert "policy_type" in data, "policy_type is required"
        if "policy_args" not in data:
            data["policy_args"] = {}

        if "num_steps" in data and data["num_steps"] is not None:
            num_steps = data["num_steps"]
        else:
            # will fall back to default num_steps from the policy or the simulation app config
            num_steps = None
        if "status" in data and data["status"] is not None:
            status = Status(data["status"])
        else:
            status = Status.PENDING

        return cls(
            name=data["name"],
            arena_env_args=cls.convert_args_dict_to_cli_args_list(data["arena_env_args"]),
            policy_type=data["policy_type"],
            num_steps=num_steps,
            policy_config_dict=data["policy_args"],
            status=status,
        )

    @classmethod
    def convert_args_dict_to_cli_args_list(cls, args_dict: dict) -> list[str]:
        """Convert a dictionary of arguments to a list of arguments that can be passed to the CLI parser.

        Args:
            args_dict: Dictionary of arguments

        Returns:
            List of arguments that can be passed to the CLI parser
        """
        args_list = []
        for key, value in args_dict.items():
            if isinstance(value, bool) and value:
                args_list += [f"--{key}"]
            elif not isinstance(value, bool) and value is not None:
                if key != "environment":
                    args_list += [f"--{key}", str(value)]
                else:
                    args_list += [str(value)]
            else:
                continue

        return args_list


class JobManager:
    def __init__(self, jobs: list[Job | dict]):
        """Initialize JobManager with a list of jobs.

        Args:
            jobs: List of Job objects or dictionaries to manage.
                  Dictionaries will be automatically converted to Job objects.
        """
        self.pending_queue = Queue()
        self.all_jobs = []  # Keep track of all jobs for status reporting

        for job_or_dict in jobs:
            if isinstance(job_or_dict, dict):
                job = Job.from_dict(job_or_dict)
            else:
                job = job_or_dict

            if job.status == Status.PENDING:
                self.pending_queue.put(job)
            self.all_jobs.append(job)

    def __iter__(self):
        return self

    def __next__(self):
        job = self.get_next_job()
        if job is not None:
            return job
        else:
            raise StopIteration

    def get_next_job(self) -> Job | None:
        """Get the next pending job from the front of the queue.

        Returns:
            Next pending Job or None if queue is empty
        """
        if not self.pending_queue.empty():
            job = self.pending_queue.get()
            job.status = Status.RUNNING
            job.start_time = time.time()
            print(f"Running job {job.name}")
            return job
        print("No pending jobs in queue")
        return None

    def set_jobs_status_by_name(self, status: Status, job_names: list[str]) -> None:
        """Set the status of jobs with the given names to the given status."""
        for job in self.all_jobs:
            if job.name in job_names:
                if job.status != Status.PENDING and status == Status.PENDING:
                    self.pending_queue.put(job)
                job.status = status

    def get_job_count(self) -> dict[Status, int]:
        """Get number of jobs grouped by status.

        Returns:
            Dictionary mapping Status to count
        """
        counts = {status.value: 0 for status in Status}
        for job in self.all_jobs:
            counts[job.status.value] += 1
        return counts

    def complete_job(self, job: Job, metrics: dict, status: Status) -> None:
        """complete a job with the given metrics and status."""
        job.metrics = metrics
        job.status = status
        job.end_time = time.time()

    def is_empty(self) -> bool:
        """Check if there are any pending jobs."""
        return self.pending_queue.empty()

    def print_jobs_info(self) -> None:
        """Print information about the jobs."""

        # print using pretty table as data fields may have various lengths
        table = PrettyTable(field_names=["Job Name", "Status", "Policy Type", "Num Steps"])
        for job in self.all_jobs:
            table.add_row([job.name, job.status.value, job.policy_type, job.num_steps])
        print(table)
