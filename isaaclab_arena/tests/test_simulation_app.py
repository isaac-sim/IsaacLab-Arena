# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from isaaclab_arena.utils.isaaclab_utils import simulation_app
from isaaclab_arena.utils.isaaclab_utils.simulation_app import _ensure_livestream_visualizer


@pytest.mark.parametrize("livestream", [1, 2])
def test_livestream_enables_kit_visualizer(livestream):
    args = argparse.Namespace(livestream=livestream, visualizer=None)

    _ensure_livestream_visualizer(args)

    assert args.visualizer == ["kit"]


def test_get_app_launcher_enables_kit_visualizer_for_livestream(monkeypatch):
    args = argparse.Namespace(livestream=1, visualizer=None)
    launched_args = None

    def fake_app_launcher(app_args):
        nonlocal launched_args
        launched_args = app_args

    monkeypatch.setattr(simulation_app, "AppLauncher", fake_app_launcher)
    simulation_app.get_app_launcher(args)

    assert launched_args.visualizer == ["kit"]


def test_livestream_environment_enables_kit_visualizer(monkeypatch):
    monkeypatch.setenv("LIVESTREAM", "1")
    args = argparse.Namespace(livestream=-1, visualizer=None)

    _ensure_livestream_visualizer(args)

    assert args.visualizer == ["kit"]


def test_livestream_preserves_explicit_visualizer():
    args = argparse.Namespace(livestream=1, visualizer=["viser"])

    _ensure_livestream_visualizer(args)

    assert args.visualizer == ["viser"]


def test_disabled_livestream_preserves_default_visualizer():
    args = argparse.Namespace(livestream=0, visualizer=None)

    _ensure_livestream_visualizer(args)

    assert args.visualizer is None
