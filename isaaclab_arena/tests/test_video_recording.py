# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for viewport video recorder cleanup."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from isaaclab_arena.video.video_recording import close_viewport_video_recorder


def test_close_viewport_video_recorder_releases_replicator_resources_once():
    """Replicator resources are detached and destroyed exactly once."""
    capture = SimpleNamespace(
        _rgb_annotator=MagicMock(),
        _render_product=MagicMock(),
    )
    recorder = SimpleNamespace(_capture=capture)
    annotator = capture._rgb_annotator
    render_product = capture._render_product

    close_viewport_video_recorder(recorder)
    close_viewport_video_recorder(recorder)

    annotator.detach.assert_called_once_with()
    render_product.destroy.assert_called_once_with()
    assert capture._rgb_annotator is None
    assert capture._render_product is None
    assert recorder._capture is None


def test_close_viewport_video_recorder_uses_public_close():
    """Recorder-owned cleanup is preferred when a public close method exists."""
    recorder = SimpleNamespace(close=MagicMock())

    close_viewport_video_recorder(recorder)

    recorder.close.assert_called_once_with()
