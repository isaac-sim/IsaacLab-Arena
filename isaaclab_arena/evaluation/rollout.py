# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Roll out policies in Arena environments."""

from __future__ import annotations

import torch
import tqdm
from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


def rollout_policy(
    env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
) -> MetricsDataCollection | None:
    """Roll out a policy until its configured step or episode limit."""
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    try:
        obs, _ = env.reset()
        policy.reset()
        policy.set_task_description(env.unwrapped.get_language_instruction())

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0

        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    # Only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    # Break if number of episodes is reached
                    completed_episodes = env_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                    if num_episodes is not None:
                        pbar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
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
