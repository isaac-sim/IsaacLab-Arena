# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Roll out a policy in an Arena environment for an evaluation run."""

from __future__ import annotations

import tqdm
from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.utils.timer import Timer

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


def rollout_policy(
    env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
) -> MetricsDataCollection | None:
    import torch

    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    try:
        if num_episodes is not None:
            env.unwrapped.configure_episode_limit(num_episodes)
        obs, _ = env.reset()
        policy.reset()
        policy.set_task_description(env.unwrapped.get_language_instruction())

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_steps_completed = 0

        while True:
            with torch.inference_mode(), Timer("step"):
                with Timer("policy_inference"):
                    actions = policy.get_action(env, obs)
                with Timer("env_step"):
                    obs, _, terminated, truncated, _ = env.step(actions)

                reset_env_ids = env.unwrapped.reset_env_ids
                if reset_env_ids.numel() > 0:
                    print(f"Resetting policy state for env_ids: {reset_env_ids}")
                    policy.reset(env_ids=reset_env_ids)

                if terminated.any() or truncated.any():
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                if num_episodes is not None:
                    completed_episode_count = env.unwrapped.completed_episode_count
                    pbar.update(completed_episode_count - pbar.n)
                    if completed_episode_count == num_episodes:
                        break
                # Break if number of steps is reached
                num_steps_completed += 1
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        pbar.close()

    except Exception as e:
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None
