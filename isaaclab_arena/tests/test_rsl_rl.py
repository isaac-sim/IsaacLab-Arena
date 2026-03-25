# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for RSL-RL: train a policy then evaluate it with the policy runner."""

import glob
import os
import time

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

NUM_STEPS = 2


def run_rl_train(
    example_environment: str,
    embodiment: str | None = None,
    object_name: str | None = None,
    num_envs: int = 1,
    max_iterations: int = 1,
) -> str:
    """Train an RSL-RL policy for a single iteration and return the checkpoint path.

    Uses IsaacLab's rsl_rl train.py with the Arena environment registration callback.
    The training script saves params/agent.yaml alongside the checkpoint, which is
    required by RslRlActionPolicy at inference time.
    """
    train_script = f"{TestConstants.submodules_dir}/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
    args = [
        TestConstants.python_path,
        train_script,
        "--external_callback",
        "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback",
        "--task",
        example_environment,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--headless",
    ]
    if embodiment is not None:
        args += ["--embodiment", embodiment]
    if object_name is not None:
        args += ["--object", object_name]

    t_start = time.time()
    run_subprocess(args)

    # Find the most recently written checkpoint produced by this training run.
    log_pattern = os.path.join(TestConstants.repo_root, "logs", "rsl_rl", "**", "*.pt")
    checkpoints = [f for f in glob.glob(log_pattern, recursive=True) if os.path.getmtime(f) >= t_start]
    if not checkpoints:
        raise RuntimeError("Training completed but no checkpoint was found under logs/rsl_rl/")
    return max(checkpoints, key=os.path.getmtime)


def run_policy_runner(checkpoint_path: str, example_environment: str, embodiment: str, object_name: str) -> None:
    args = [
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/policy_runner.py",
        "--policy_type",
        "rsl_rl",
        "--checkpoint_path",
        checkpoint_path,
        "--num_steps",
        str(NUM_STEPS),
        "--headless",
        example_environment,
        "--embodiment",
        embodiment,
        "--object",
        object_name,
    ]
    run_subprocess(args)


@pytest.mark.with_subprocess
@pytest.mark.skip(reason="Broken. Waiting for Isaac Lab 3.0 version of interop to be merged.")
def test_rl_train_and_eval_lift_object():
    checkpoint_path = run_rl_train(
        example_environment="lift_object",
        embodiment="franka",
        object_name="dex_cube",
    )
    run_policy_runner(
        checkpoint_path=checkpoint_path,
        example_environment="lift_object",
        embodiment="franka",
        object_name="dex_cube",
    )
