# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for single-arm OpenXR CLI selection and retargeting pipeline wiring."""

import argparse

from isaaclab_arena.teleop import enable_openxr_teleop_from_cli


def test_xr_selects_openxr_teleop_device():
    args = argparse.Namespace(xr=True, teleop_device=None)

    enable_openxr_teleop_from_cli(args)

    assert args.xr is True
    assert args.teleop_device == "openxr"


def test_openxr_teleop_device_enables_xr():
    args = argparse.Namespace(xr=False, teleop_device="openxr")

    enable_openxr_teleop_from_cli(args)

    assert args.xr is True
    assert args.teleop_device == "openxr"


def test_explicit_non_xr_device_is_preserved():
    args = argparse.Namespace(xr=True, teleop_device="keyboard")

    enable_openxr_teleop_from_cli(args)

    assert args.xr is True
    assert args.teleop_device == "keyboard"


def test_single_arm_openxr_pipeline_has_seven_dimensional_action():
    from isaaclab_arena.teleop.single_arm_openxr_pipeline import build_single_arm_openxr_pipeline

    output_type = build_single_arm_openxr_pipeline().output_types()["action"]
    action_array_type = output_type._types[0]  # noqa: SLF001

    assert action_array_type.shape == (7,)
