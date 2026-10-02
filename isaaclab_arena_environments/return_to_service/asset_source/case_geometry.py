# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dimensioned case shells, passive hinge hardware and local grasp frames."""

from __future__ import annotations

import math

from .geometry import Geometry

AXIS_X = (0, math.pi / 2, 0)


def _latch_grasp() -> dict:
    """Return the configured DROID TCP pose in the latch pivot frame."""
    tilt = math.radians(15)
    half_cosine = math.cos(tilt / 2)
    half_sine = math.sin(tilt / 2)
    lower = (half_cosine - half_sine) / 2
    upper = (half_cosine + half_sine) / 2
    # Contact lies 29 mm along tool +X from its TCP. The tilt keeps the
    # configured fingers clear of the hook and its mounting hardware.
    return {
        "position_xyz": [0, -0.025 + 0.029 * math.sin(tilt), 0.090 + 0.029 * math.cos(tilt)],
        "contact_center_xyz": [0, -0.025, 0.090],
        "rotation_xyzw": [lower, lower, -upper, upper],
        "jaw_span": 0.046,
        "frame": "case_latch",
        "approach_offset_case_xyz": [0, 0, 0.100],
    }


def _build_base(g: Geometry) -> None:
    """Build the lower shell and external supports for the latch pivot."""
    g.asset(
        "case_base",
        0.48,
        interior_bounds=[[-0.184, -0.121, 0.008], [0.184, 0.121, 0.162]],
        lid_closed_pose={"position_xyz": [0, 0.1325, 0.035], "rotation_xyzw": [0, 0, 0, 1]},
        hinge={"axis": [1, 0, 0], "open_angle": -1.8, "angle_limits_degrees": [-110, 0]},
        latch={
            "position_xyz": [0, -0.220, 0.140],
            "axis": [1, 0, 0],
            "angle_limits_degrees": [0, 100],
            "static_friction_effort_nm": 0.012,
            "dynamic_friction_effort_nm": 0.010,
        },
        lid_handle_grasp={
            "position_xyz": [0.160, -0.305, 0.075],
            "rotation_xyzw": [0.5, 0.5, -0.5, 0.5],
            "jaw_span": 0.030,
            "frame": "case_lid",
        },
        latch_grasp=_latch_grasp(),
        packing_poses={
            "body": [-0.010, -0.040, 0.011],
            "battery": [-0.120, 0.075, 0.011],
            "crevice_tool": [-0.007, 0.075, 0.011],
            "brush_tool": [0.121, 0.075, 0.011],
        },
    )
    g.box("lower_shell", (0, 0, 0.004), (0.390, 0.265, 0.008), "navy", 0.004, collision=True)
    for side in (-1, 1):
        g.box("side_wall", (side * 0.190, 0, 0.021), (0.010, 0.265, 0.034), "navy", 0.004, collision=True)
        g.box("end_wall", (0, side * 0.1275, 0.021), (0.37, 0.010, 0.034), "navy", 0.004, collision=True)
        g.cylinder("hinge_knuckle", (side * 0.125, 0.1325, 0.035), 0.009, 0.060, "steel", AXIS_X)
        g.box("rubber_foot", (side * 0.155, -0.090, -0.002), (0.026, 0.022, 0.005), "rubber", 0.002)
    g.box("foam_liner", (0, 0, 0.0085), (0.367, 0.242, 0.003), "foam", 0.002)
    g.box("component_separator", (0, 0.022, 0.018), (0.350, 0.006, 0.023), "foam", 0.002, collision=True)
    g.label("TESTED UNIT", (0, -0.106, 0.011), 0.012, "ivory")
    g.label("BATTERY", (-0.111, 0.106, 0.011), 0.007, "ivory")
    g.label("TOOLS", (0.078, 0.106, 0.011), 0.007, "ivory")
    # Folded-steel supports leave the central lever's entire travel clear.
    for side in (-1, 1):
        x = side * 0.045
        g.box("latch_mount_foot", (x, -0.141, 0.021), (0.018, 0.025, 0.022), "steel", 0.002, collision=True)
        start, end = (x, -0.146, 0.034), (x, -0.220, 0.128)
        length = math.hypot(end[1] - start[1], end[2] - start[2])
        angle = -math.atan2(end[1] - start[1], end[2] - start[2])
        center = tuple((a + b) / 2 for a, b in zip(start, end))
        g.box(
            "latch_mount_strut", center, (0.012, 0.012, length), "steel", 0.002, collision=True, rotation=(angle, 0, 0)
        )
        g.box("latch_pivot_ear", (side * 0.0275, -0.220, 0.140), (0.008, 0.020, 0.024), "steel", 0.002, collision=True)
        g.box(
            "latch_mount_bridge", (side * 0.037, -0.220, 0.128), (0.026, 0.016, 0.008), "steel", 0.001, collision=True
        )
        g.cylinder("latch_mount_fastener", (x, -0.154, 0.026), 0.004, 0.002, "steel", (math.pi / 2, 0, 0))


def _build_lid(g: Geometry) -> None:
    """Build the hinge-origin upper shell, exterior pull and raised catch."""
    g.asset("case_lid", 0.38, origin="hinge")
    g.box("upper_shell", (0, -0.1325, 0.131), (0.390, 0.265, 0.008), "navy", 0.004, collision=True)
    for side in (-1, 1):
        g.box("lid_side", (side * 0.191, -0.1325, 0.0675), (0.008, 0.265, 0.135), "navy", 0.004, collision=True)
        g.box("lid_end", (0, -0.1325 + side * 0.1285, 0.0675), (0.374, 0.008, 0.135), "navy", 0.004, collision=True)
        g.box("stacking_rail", (side * 0.120, -0.1325, 0.138), (0.022, 0.223, 0.007), "rubber", 0.003)
        g.cylinder("hinge_knuckle", (side * 0.066, 0, 0), 0.009, 0.048, "steel", AXIS_X)
    g.box("brand_panel", (0, -0.1325, 0.136), (0.192, 0.110, 0.003), "teal", 0.006)
    g.label("RETURN TO SERVICE", (0, -0.117, 0.138), 0.014)
    g.label("CIRCULAR EQUIPMENT / A", (0, -0.149, 0.138), 0.008)
    # A front-facing vertical pull keeps the gripper palm outside the lid's
    # falling sweep. Its offset from the center leaves the latch accessible.
    g.box("closing_grip", (0.160, -0.305, 0.075), (0.030, 0.022, 0.100), "rubber", 0.005, collision=True)
    for z in (0.031, 0.119):
        g.box("closing_grip_mount", (0.160, -0.283, z), (0.030, 0.044, 0.012), "steel", 0.002, collision=True)
    # The keeper is above the front lip and outside the packing volumes.
    g.box("catch_mount", (0, -0.2725, 0.139), (0.036, 0.014, 0.044), "steel", 0.002, collision=True)
    g.box("catch_mount_foot", (0, -0.263, 0.128), (0.052, 0.028, 0.008), "steel", 0.002, collision=True)
    g.box("catch", (0, -0.2775, 0.161), (0.036, 0.012, 0.012), "steel", 0.002, collision=True)
    for side in (-1, 1):
        g.cylinder("catch_fastener", (side * 0.018, -0.263, 0.133), 0.003, 0.002, "steel")


def _build_latch(g: Geometry) -> None:
    """Build the 45 gram rotary lever, gripping flange and passive friction rivet."""
    g.asset("case_latch", 0.045, origin="pivot", axis=[1, 0, 0])
    g.cylinder("pivot", (0, 0, 0), 0.0045, 0.062, "steel", AXIS_X, collision=True)
    for side in (-1, 1):
        g.ring("friction_washer", (side * 0.0195, 0, 0), 0.0075, 0.0045, 0.002, "rubber", AXIS_X)
        g.cylinder("rivet_head", (side * 0.032, 0, 0), 0.0055, 0.002, "steel", AXIS_X)
    g.box(
        "lever",
        (0, 0.0225, 0.032),
        (0.018, 0.010, math.hypot(0.045, 0.064)),
        "orange",
        0.002,
        collision=True,
        rotation=(-math.atan2(0.045, 0.064), 0, 0),
    )
    g.box("hook", (0, 0.061, 0.066), (0.035, 0.034, 0.007), "steel", 0.002, collision=True)
    g.box(
        "handle_stem",
        (0, -0.0125, 0.041),
        (0.012, 0.012, math.hypot(0.025, 0.082)),
        "orange",
        0.002,
        collision=True,
        rotation=(math.atan2(0.025, 0.082), 0, 0),
    )
    g.box("grasp_flange", (0, -0.025, 0.090), (0.046, 0.022, 0.020), "orange", 0.003, collision=True)
    g.box("grip_inlay", (0, -0.0363, 0.090), (0.030, 0.001, 0.014), "rubber", 0.002)


def build_case(g: Geometry) -> None:
    """Build the hollow case as three independently exported rigid parts.

    Args:
        g: Asset authoring context receiving the base, lid and latch.
    """
    _build_base(g)
    _build_lid(g)
    _build_latch(g)
