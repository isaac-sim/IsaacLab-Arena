# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release resources owned by an Arena evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    collect_garbage_and_clear_cuda_cache,
    teardown_simulation_app,
)

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.policy.policy_base import PolicyBase


def close_policy(policy: PolicyBase | None) -> None:
    """Close a policy and release cached runtime resources."""
    try:
        if policy is not None:
            policy.close()
    finally:
        collect_garbage_and_clear_cuda_cache()


def close_environment(env: gym.Env | None) -> None:
    """Tear down and close an instantiated environment."""
    if env is None:
        return
    try:
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
    finally:
        try:
            env.close()
        finally:
            collect_garbage_and_clear_cuda_cache()


def close_experiment_resources(policy: PolicyBase | None, env: gym.Env | None) -> None:
    """Close an experiment's policy and environment."""
    try:
        close_policy(policy)
    finally:
        close_environment(env)
