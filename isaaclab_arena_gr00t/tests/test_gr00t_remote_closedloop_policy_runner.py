# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for policy_runner with a live GR00T remote closed-loop server.

This test intentionally checks runner plumbing, not task success. CI starts a
GR00T sidecar service, waits until it answers ping, then runs this test against
that service. Any runner, env, camera, RPC, observation conversion, or action
conversion exception should surface as a nonzero subprocess exit.
"""

import os

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

pytestmark = [
    pytest.mark.with_cameras,
    pytest.mark.with_subprocess,
    pytest.mark.gr00t_remote_e2e,
]

REMOTE_HOST_ENV = "ISAACLAB_ARENA_GR00T_REMOTE_HOST"
REMOTE_PORT_ENV = "ISAACLAB_ARENA_GR00T_REMOTE_PORT"
TIMEOUT_ENV = "ISAACLAB_ARENA_GR00T_REMOTE_E2E_TIMEOUT"

POLICY_TYPE = "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml"
ENVIRONMENT = "galileo_g1_locomanip_pick_and_place"
OBJECT_NAME = "brown_box"
EMBODIMENT = "g1_wbc_joint"
NUM_STEPS = 17
NUM_ENVS = 1


def test_policy_runner_with_gr00t_remote_closedloop_policy():
    """Run a short closed-loop rollout against the live GR00T server."""
    remote_host = os.environ.get(REMOTE_HOST_ENV)
    if not remote_host:
        pytest.skip(f"Set {REMOTE_HOST_ENV} to run this test against a live GR00T policy server.")

    remote_port = int(os.environ.get(REMOTE_PORT_ENV, "5555"))
    timeout_sec = int(os.environ.get(TIMEOUT_ENV, "900"))

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
        str(NUM_ENVS),
        "--headless",
        "--enable_cameras",
        ENVIRONMENT,
        "--object",
        OBJECT_NAME,
        "--embodiment",
        EMBODIMENT,
    ]

    run_subprocess(args, timeout_sec=timeout_sec)
