# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run an Arena policy against an instantiated environment."""

from __future__ import annotations

import torch
import tqdm
from typing import TYPE_CHECKING

from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


def rollout_policy(
    env: gym.Env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
) -> MetricsDataCollection | None:
    """Roll out a policy until its configured step or episode limit is reached."""
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    progress_bar = None
    try:
        observation, _ = env.reset()
        policy.reset()
        policy.set_task_description(env.unwrapped.get_language_instruction())

        if num_steps is not None:
            progress_bar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            progress_bar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0

        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, observation)
                observation, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    completed_episodes = env_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                    if num_episodes is not None:
                        progress_bar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
                            break

                num_steps_completed += 1
                if num_steps is not None:
                    progress_bar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        progress_bar.close()

    except Exception as error:
        if progress_bar is not None:
            progress_bar.close()
        raise RuntimeError(f"Error rolling out policy: {error}") from error

    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
        return env.unwrapped.compute_metrics()
    return None
