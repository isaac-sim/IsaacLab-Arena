# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.evaluation.job_manager import Job, JobManager, Status


def test_job_from_dict():
    """Test creating a Job from a dictionary."""
    job_dict = {
        "name": "test_job_dict",
        "arena_env_args": {"environment": "env2", "arg2": "value2"},
        "policy_type": "g00t",
        "num_steps": 50,
        "policy_args": {"policy_device": "cpu"},
        "status": "completed",
    }

    job = Job.from_dict(job_dict)

    assert job.name == "test_job_dict"
    # Dict args get converted to CLI args list
    assert "env2" in job.arena_env_args
    assert "--arg2" in job.arena_env_args
    assert job.policy_type == "g00t"
    assert job.num_steps == 50
    assert "--policy_device" in job.policy_args
    assert "cpu" in job.policy_args
    assert job.status == Status.COMPLETED


def test_job_convert_args_dict_to_cli_args_list():
    """Test converting arguments dictionary to CLI args list."""
    # Test basic string arguments
    args_dict = {"env": "test_env", "object": "box", "num_envs": "4"}
    args_list = Job.convert_args_dict_to_cli_args_list(args_dict)
    assert "--env" in args_list
    assert "test_env" in args_list
    assert "--object" in args_list
    assert "box" in args_list
    assert "--num_envs" in args_list
    assert "4" in args_list

    # Test boolean arguments
    args_dict_bool = {"headless": True, "enable_cameras": False}
    args_list_bool = Job.convert_args_dict_to_cli_args_list(args_dict_bool)
    assert "--headless" in args_list_bool
    assert "--enable_cameras" not in args_list_bool  # False booleans are skipped

    # Test empty dict
    assert Job.convert_args_dict_to_cli_args_list({}) == []


def test_job_manager_update_job_status():
    """Test updating a job status."""

    job = Job("test_job", {"environment": "test_env"}, "zero_action", policy_args={})
    job_manager = JobManager([job])

    # Get the job
    job = job_manager.get_next_job()
    assert job.status == Status.RUNNING

    counts = job_manager.get_job_count()
    assert counts["pending"] == 0
    assert counts["running"] == 1
    assert counts["completed"] == 0
    assert counts["failed"] == 0


def test_job_manager_failed_job():
    """Test marking a job as failed."""
    import time

    job = Job("failing_job", {"environment": "test_env"}, "zero_action", policy_args={})
    job_manager = JobManager([job])

    job = job_manager.get_next_job()
    # Manually mark as failed
    job.status = Status.FAILED
    job.end_time = time.time()
    job.metrics = {}

    assert job.status == Status.FAILED
    assert job.metrics == {}

    counts = job_manager.get_job_count()
    assert counts["failed"] == 1


def test_job_manager_mixed_statuses():
    """Test JobManager with jobs in various states."""
    jobs = [
        {
            "name": "pending_job",
            "arena_env_args": {"environment": "env1"},
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "completed_job",
            "arena_env_args": {"environment": "env2"},
            "policy_type": "random",
            "policy_args": {},
            "status": "completed",
        },
        {
            "name": "failed_job",
            "arena_env_args": {"environment": "env3"},
            "policy_type": "random",
            "policy_args": {},
            "status": "failed",
        },
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
        {"name": "first", "arena_env_args": {"env": "env1"}, "policy_type": "zero_action", "policy_args": {}},
        {"name": "second", "arena_env_args": {"env": "env2"}, "policy_type": "random", "policy_args": {}},
        {"name": "third", "arena_env_args": {"env": "env3"}, "policy_type": "zero_action", "policy_args": {}},
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


def test_job_manager_set_jobs_status_by_name():
    """Test setting job statuses by name."""
    import time

    jobs = [
        {
            "name": "job1",
            "arena_env_args": {"environment": "env1"},
            "policy_type": "zero_action",
            "policy_args": {},
            "num_steps": 10,
        },
        {
            "name": "job2",
            "arena_env_args": {"environment": "env2"},
            "policy_type": "random",
            "policy_args": {},
            "num_steps": 20,
        },
    ]

    job_manager = JobManager(jobs)

    # Get first job and mark it as completed
    job1 = job_manager.get_next_job()
    job1.status = Status.COMPLETED
    job1.end_time = time.time()

    # Get second job and mark it as failed
    job2 = job_manager.get_next_job()
    job2.status = Status.FAILED
    job2.end_time = time.time()

    # Verify status
    counts = job_manager.get_job_count()
    assert counts["completed"] == 1
    assert counts["failed"] == 1
    assert counts["pending"] == 0

    # Reset job1 back to pending
    job_manager.set_jobs_status_by_name(Status.PENDING, ["job1"])

    # Verify job1 is back in pending queue
    counts = job_manager.get_job_count()
    assert counts["pending"] == 1
    assert counts["completed"] == 0
    assert counts["failed"] == 1
    assert not job_manager.is_empty()

    # Get the job again
    job = job_manager.get_next_job()
    assert job.name == "job1"


def test_job_manager_iterator():
    """Test that JobManager is iterable."""
    jobs = [
        {"name": "job1", "arena_env_args": {"env": "env1"}, "policy_type": "zero_action", "policy_args": {}},
        {"name": "job2", "arena_env_args": {"env": "env2"}, "policy_type": "random", "policy_args": {}},
        {"name": "job3", "arena_env_args": {"env": "env3"}, "policy_type": "zero_action", "policy_args": {}},
    ]

    job_manager = JobManager(jobs)

    # Iterate through jobs using for loop
    job_names = []
    for job in job_manager:
        job_names.append(job.name)
        assert job.status == Status.RUNNING  # Each job should be running when yielded

    assert job_names == ["job1", "job2", "job3"]
    assert job_manager.is_empty()
