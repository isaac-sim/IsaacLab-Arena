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
        arena_env_args: list[str],
        policy_type: str,
        num_steps: int = None,
        policy_args: list[str] = [],
        status: Status = Status.PENDING,
    ):
        """Initialize a Job instance.

        Args:
            name: Job name, used to identify the job in the queue and in the logs.
            arena_env_args: List of CLI arguments for configuring the arena environment
            num_steps: Number of steps to run the policy for
            policy_type: Type of policy to use
            policy_args: List of CLI arguments for the policy. These are passed to the policy class's from_args method.
            status: Job status (defaults to PENDING)
        """
        self.name = name
        self.arena_env_args = arena_env_args
        self.num_steps = num_steps
        self.policy_type = policy_type
        self.policy_args = policy_args
        self.status = status
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create a Job instance from a dictionary.

        Args:
            data: Dictionary containing job data with keys:
                  - name: Job name
                  - arena_env_args: List of CLI arguments for configuring the arena environment
                  - num_steps: Number of steps to run the policy for
                  - policy_type: Type of policy to use
                  - policy_args: List of CLI arguments for the policy. These are passed to the policy class's from_args method.
                  - status: Status string (optional, defaults to PENDING)

        Returns:
            New Job instance
        """
        name = data["name"]
        arena_env_args = data["arena_env_args"]
        policy_type = data["policy_type"]
        policy_args = data.get("policy_args", [])
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
            name=name,
            arena_env_args=arena_env_args,
            policy_type=policy_type,
            num_steps=num_steps,
            policy_args=policy_args,
            status=status,
        )


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

    def complete_job(self, job: Job, metrics: dict[str, float], status: Status):
        """Complete a job and store the metrics.

        Args:
            job: The job to complete.
            metrics: The metrics to store.
            status: The status of the job.
        """
        job.status = status
        job.end_time = time.time()
        job.metrics = metrics
        print(f"Job {job.name} {status.value}")

    def get_job_count(self) -> dict[Status, int]:
        """Get number of jobs grouped by status.

        Returns:
            Dictionary mapping Status to count
        """
        counts = {status.value: 0 for status in Status}
        for job in self.all_jobs:
            counts[job.status.value] += 1
        return counts

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
