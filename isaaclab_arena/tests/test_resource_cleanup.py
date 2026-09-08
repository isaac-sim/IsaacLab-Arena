# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluation resource cleanup ordering."""

from types import SimpleNamespace

from isaaclab_arena.evaluation import resource_cleanup


def test_close_environment_releases_viewport_before_replacing_stage(monkeypatch):
    calls = []
    video_recorder = object()
    env = SimpleNamespace(
        close=lambda: calls.append("env.close"),
        unwrapped=SimpleNamespace(video_recorder=video_recorder),
    )

    monkeypatch.setattr(
        resource_cleanup,
        "close_viewport_video_recorder",
        lambda recorder: calls.append(("close_viewport_video_recorder", recorder)),
    )
    monkeypatch.setattr(
        resource_cleanup,
        "teardown_simulation_app",
        lambda **kwargs: calls.append(("teardown_simulation_app", kwargs)),
    )
    monkeypatch.setattr(
        resource_cleanup,
        "collect_garbage_and_clear_cuda_cache",
        lambda: calls.append("collect_garbage_and_clear_cuda_cache"),
    )

    resource_cleanup.close_environment(env)

    assert calls == [
        ("close_viewport_video_recorder", video_recorder),
        (
            "teardown_simulation_app",
            {"suppress_exceptions": False, "make_new_stage": True},
        ),
        "env.close",
        "collect_garbage_and_clear_cuda_cache",
    ]
