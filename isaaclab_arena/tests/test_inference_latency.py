# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the inference latency model: step conversion and the per-step action delay."""

import torch

import pytest

NUM_ENVS = 3
ACTION_DIM = 2
CONTROL_STEP_DT = 1.0 / 15.0


@pytest.mark.parametrize(
    ("latency_ms", "expected_steps"),
    [
        (0.0, 0),
        (1.0, 1),
        (1000.0 / 15.0, 1),  # exactly one control step
        (85.0, 2),
        (150.0, 3),
        (1000.0, 15),
    ],
)
def test_inference_latency_rounds_up_to_whole_control_steps(latency_ms, expected_steps):
    from isaaclab_arena.policy.inference_latency import inference_latency_steps

    assert inference_latency_steps(latency_ms, CONTROL_STEP_DT) == expected_steps


def test_inference_latency_rejects_negative_values():
    from isaaclab_arena.policy.inference_latency import inference_latency_steps

    with pytest.raises(AssertionError, match="non-negative"):
        inference_latency_steps(-1.0, CONTROL_STEP_DT)


def _step_action(step: int) -> torch.Tensor:
    return torch.full((NUM_ENVS, ACTION_DIM), float(step))


def test_per_step_latency_holds_then_returns_the_action_from_d_steps_earlier():
    from isaaclab_arena.policy.inference_latency import PerStepInferenceLatency

    latency = PerStepInferenceLatency(NUM_ENVS, latency_steps=2, device="cpu")
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    applied = [latency.delay(_step_action(step), hold)[:, 0].tolist() for step in range(5)]

    assert applied == [[-1.0] * NUM_ENVS, [-1.0] * NUM_ENVS, [0.0] * NUM_ENVS, [1.0] * NUM_ENVS, [2.0] * NUM_ENVS]
    assert latency.hold_steps.tolist() == [2, 2, 2]


def test_per_step_latency_reset_restarts_the_hold_for_the_reset_envs_only():
    from isaaclab_arena.policy.inference_latency import PerStepInferenceLatency

    latency = PerStepInferenceLatency(NUM_ENVS, latency_steps=1, device="cpu")
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)
    latency.delay(_step_action(0), hold)
    latency.delay(_step_action(1), hold)

    latency.reset(torch.tensor([1]))
    applied = latency.delay(_step_action(2), hold)[:, 0].tolist()

    assert applied == [1.0, -1.0, 1.0]
