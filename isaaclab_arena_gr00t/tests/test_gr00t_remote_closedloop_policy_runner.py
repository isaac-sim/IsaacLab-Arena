# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests with a live GR00T remote closed-loop server.

This test intentionally checks runner plumbing, not task success. CI starts a
GR00T sidecar service, waits until it answers ping, then runs these tests
against that service. Any runner, env, camera, RPC, observation conversion, or
action conversion exception should surface as a nonzero subprocess exit.
"""

import json
import os

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

pytestmark = [
    pytest.mark.with_cameras,
    pytest.mark.with_subprocess,
    pytest.mark.gr00t_remote_e2e,
]

REMOTE_HOST_ENV = "GR00T_REMOTE_HOST"
REMOTE_PORT_ENV = "GR00T_REMOTE_PORT"
TIMEOUT_ENV = "GR00T_REMOTE_E2E_TIMEOUT"
DEFAULT_TIMEOUT_SEC = 900

POLICY_TYPE = "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml"
ENVIRONMENT = "galileo_g1_locomanip_pick_and_place"
OBJECT_NAME = "brown_box"
EMBODIMENT = "g1_wbc_joint"
NUM_STEPS = 17
SINGLE_ENV_COUNT = 1
MULTI_ENV_COUNT = 3
EXPERIMENT_RUNNER_NUM_STEPS = 2


def _get_gr00t_remote_server() -> tuple[str, int, int]:
    """Return live GR00T server settings or skip when no sidecar is configured."""
    remote_host = os.environ.get(REMOTE_HOST_ENV)
    if not remote_host:
        pytest.skip(f"Set {REMOTE_HOST_ENV} to run this test against a live GR00T policy server.")

    remote_port = int(os.environ.get(REMOTE_PORT_ENV, "5555"))
    timeout_sec = int(os.environ.get(TIMEOUT_ENV, str(DEFAULT_TIMEOUT_SEC)))
    return remote_host, remote_port, timeout_sec


def _run_policy_runner(remote_host: str, remote_port: int, timeout_sec: int, num_envs: int) -> None:
    args = [
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/policy_runner.py",
        "--policy_type",
        POLICY_TYPE,
        "--policy_config_yaml_path",
        POLICY_CONFIG,
        "--remote_host",
        remote_host,
        "--remote_port",
        str(remote_port),
        "--num_steps",
        str(NUM_STEPS),
        "--num_envs",
        str(num_envs),
        "--headless",
        "--enable_cameras",
        ENVIRONMENT,
        "--object",
        OBJECT_NAME,
        "--embodiment",
        EMBODIMENT,
    ]

    run_subprocess(args, timeout_sec=timeout_sec)


def _write_eval_jobs_config(tmp_path, remote_host: str, remote_port: int) -> str:
    jobs = [
        {
            "name": "g1_locomanip_gr00t_remote_closedloop",
            "arena_env_args": {
                "enable_cameras": True,
                "environment": ENVIRONMENT,
                "object": OBJECT_NAME,
                "embodiment": EMBODIMENT,
            },
            "num_steps": EXPERIMENT_RUNNER_NUM_STEPS,
            "policy_type": POLICY_TYPE,
            "policy_config_dict": {
                "policy_config_yaml_path": POLICY_CONFIG,
                "policy_device": "cuda",
                "remote_host": remote_host,
                "remote_port": remote_port,
            },
        },
    ]
    config_path = tmp_path / "test_gr00t_remote_closedloop_experiment_runner.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=4)
    return str(config_path)


def test_policy_runner_with_gr00t_remote_closedloop_policy_single_env():
    """Run a short single-env closed-loop rollout against the live GR00T server."""
    remote_host, remote_port, timeout_sec = _get_gr00t_remote_server()
    _run_policy_runner(remote_host, remote_port, timeout_sec, num_envs=SINGLE_ENV_COUNT)


def test_policy_runner_with_gr00t_remote_closedloop_policy_multi_env():
    """Run a short multi-env closed-loop rollout against the live GR00T server."""
    remote_host, remote_port, timeout_sec = _get_gr00t_remote_server()
    _run_policy_runner(remote_host, remote_port, timeout_sec, num_envs=MULTI_ENV_COUNT)


def test_experiment_runner_with_gr00t_remote_closedloop_policy(tmp_path):
    """Run the Experiment Runner with the GR00T remote closed-loop policy."""
    remote_host, remote_port, timeout_sec = _get_gr00t_remote_server()
    config_path = _write_eval_jobs_config(tmp_path, remote_host, remote_port)

    args = [
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/experiment_runner.py",
        "--eval_jobs_config",
        config_path,
        "--headless",
    ]
    run_subprocess(args, timeout_sec=timeout_sec)
