# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Callable, Sequence
from copy import deepcopy


def step_to_task_success(
    env,
    expected_steps: int,
    before_step: Callable[[], None] | None = None,
    expected_env_ids: Sequence[int] | None = None,
) -> dict:
    """Assert success on the specified step and return the completed progress snapshot."""
    expected_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success_env_ids = list(range(env.num_envs)) if expected_env_ids is None else list(expected_env_ids)
    expected_success[success_env_ids] = True
    assert expected_steps > 0
    assert success_env_ids

    with torch.inference_mode():
        for step_index in range(expected_steps):
            if before_step is not None:
                before_step()
            actions = torch.zeros(env.action_space.shape, device=env.device)
            _, _, terminated, truncated, extras = env.step(actions)
            assert not bool(truncated.any()), "The task timed out before the expected success."
            expected_terminated = (
                expected_success if step_index == expected_steps - 1 else torch.zeros_like(expected_success)
            )
            torch.testing.assert_close(terminated, expected_terminated)

    progress = deepcopy(extras["progress_tracking"])
    for env_id in success_env_ids:
        assert progress["states"][env_id].all_complete
        assert progress["events"][env_id], "The successful episode has no completion events."
    return progress
