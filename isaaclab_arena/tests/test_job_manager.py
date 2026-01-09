# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status


def test_job_creation():
    """Test creating a Job directly."""
    job = Job(
        name="test_job",
        arena_env_args=["env1", "--arg1", "value1"],
        policy_type="zero_action",
        num_steps=100,
        policy_args=["--policy_arg", "value"],
    )

    assert job.name == "test_job"
    assert job.arena_env_args == ["env1", "--arg1", "value1"]
    assert job.policy_type == "zero_action"
    assert job.num_steps == 100
    assert job.policy_args == ["--policy_arg", "value"]
    assert job.status == Status.PENDING
    assert job.start_time is None
    assert job.end_time is None
    assert job.metrics == {}


def test_job_from_dict():
    """Test creating a Job from a dictionary."""
    job_dict = {
        "name": "test_job_dict",
        "arena_env_args": ["env2", "--arg2"],
        "policy_type": "random",
        "num_steps": 50,
        "policy_args": ["--policy_device", "cpu"],
        "status": Status.COMPLETED,
    }

    job = Job.from_dict(job_dict)

    assert job.name == "test_job_dict"
    assert job.arena_env_args == ["env2", "--arg2"]
    assert job.policy_type == "random"
    assert job.num_steps == 50
    assert job.policy_args == ["--policy_device", "cpu"]
    assert job.status == Status.COMPLETED


def test_job_manager_initialization_with_dicts():
    """Test initializing JobManager with job dictionaries."""
    jobs_data = [
        {
            "name": "job1",
            "arena_env_args": ["env1"],
            "policy_type": "zero_action",
            "num_steps": 10,
        },
        {
            "name": "job2",
            "arena_env_args": ["env2"],
            "policy_type": "random",
            "num_steps": 20,
        },
    ]

    job_manager = JobManager(jobs_data)

    assert len(job_manager.all_jobs) == 2
    assert not job_manager.is_empty()

    counts = job_manager.get_job_count()
    assert counts["pending"] == 2
    assert counts["running"] == 0
    assert counts["completed"] == 0
    assert counts["failed"] == 0


def test_job_manager_initialization_with_job_objects():
    """Test initializing JobManager with Job objects."""
    job1 = Job("job1", ["env1"], "zero_action", num_steps=10)
    job2 = Job("job2", ["env2"], "random", num_steps=20)

    job_manager = JobManager([job1, job2])

    assert len(job_manager.all_jobs) == 2
    assert not job_manager.is_empty()


def test_job_manager_get_next_job():
    """Test getting the next job from the queue."""
    jobs_data = [
        {"name": "job1", "arena_env_args": ["env1"], "policy_type": "zero_action"},
        {"name": "job2", "arena_env_args": ["env2"], "policy_type": "random"},
    ]

    job_manager = JobManager(jobs_data)

    # Get first job
    job1 = job_manager.get_next_job()
    assert job1 is not None
    assert job1.name == "job1"
    assert job1.status == Status.RUNNING
    assert job1.start_time is not None

    # Get second job
    job2 = job_manager.get_next_job()
    assert job2 is not None
    assert job2.name == "job2"
    assert job2.status == Status.RUNNING

    # Queue should be empty now
    assert job_manager.is_empty()
    job3 = job_manager.get_next_job()
    assert job3 is None


def test_job_manager_complete_job():
    """Test completing a job with metrics."""
    job = Job("test_job", ["env"], "zero_action")
    job_manager = JobManager([job])

    # Get and complete the job
    job = job_manager.get_next_job()
    assert job.status == Status.RUNNING

    metrics = {"success_rate": 0.95, "avg_reward": 100.5}
    job_manager.complete_job(job, metrics=metrics, status=Status.COMPLETED)

    assert job.status == Status.COMPLETED
    assert job.end_time is not None
    assert job.metrics == metrics

    counts = job_manager.get_job_count()
    assert counts["pending"] == 0
    assert counts["running"] == 0
    assert counts["completed"] == 1
    assert counts["failed"] == 0


def test_job_manager_failed_job():
    """Test marking a job as failed."""
    job = Job("failing_job", ["env"], "zero_action")
    job_manager = JobManager([job])

    job = job_manager.get_next_job()
    job_manager.complete_job(job, metrics={}, status=Status.FAILED)

    assert job.status == Status.FAILED
    assert job.metrics == {}

    counts = job_manager.get_job_count()
    assert counts["failed"] == 1


def test_job_manager_mixed_statuses():
    """Test JobManager with jobs in various states."""
    jobs = [
        {"name": "pending_job", "arena_env_args": ["env1"], "policy_type": "zero_action"},
        {"name": "completed_job", "arena_env_args": ["env2"], "policy_type": "random", "status": "completed"},
        {"name": "failed_job", "arena_env_args": ["env3"], "policy_type": "random", "status": "failed"},
    ]

    job_manager = JobManager(jobs)

    # Only pending job should be in queue
    assert not job_manager.is_empty()

    counts = job_manager.get_job_count()
    assert counts["pending"] == 1
    assert counts["completed"] == 1
    assert counts["failed"] == 1

    # Get the pending job
    job = job_manager.get_next_job()
    assert job.name == "pending_job"
    assert job_manager.is_empty()


def test_job_manager_job_order():
    """Test that jobs are processed in FIFO order."""
    jobs = [
        {"name": "first", "arena_env_args": ["env1"], "policy_type": "zero_action"},
        {"name": "second", "arena_env_args": ["env2"], "policy_type": "random"},
        {"name": "third", "arena_env_args": ["env3"], "policy_type": "zero_action"},
    ]

    job_manager = JobManager(jobs)

    job1 = job_manager.get_next_job()
    job2 = job_manager.get_next_job()
    job3 = job_manager.get_next_job()

    assert job1.name == "first"
    assert job2.name == "second"
    assert job3.name == "third"


def test_job_manager_empty_initialization():
    """Test initializing JobManager with no jobs."""
    job_manager = JobManager([])

    assert len(job_manager.all_jobs) == 0
    assert job_manager.is_empty()

    job = job_manager.get_next_job()
    assert job is None