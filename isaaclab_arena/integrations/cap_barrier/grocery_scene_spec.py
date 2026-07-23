# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight constants for the calibrated CAP grocery-to-bin scene."""

CAP_GROCERY_OBJECT_ASSET = "alphabet_soup_can_hope_robolab"
CAP_GROCERY_BIN_ASSET = "grey_bin_robolab"
CAP_GROCERY_SUPPORT_ASSET = "procedural_table"
CAP_GROCERY_SUPPORT_INSTANCE = "table"
CAP_GROCERY_CAMERA_NAME = "exterior_cam"
CAP_GROCERY_CAMERA_PROFILES = ("libero", "oblique")

# Exact initial poses from the successful DROID rollout recorded in
# Isaac-cap/docs/evidence/single_object_uv_rebase_seed71.jsonl. The CAP
# arena_droid_b1 calibration pins T_world_base to identity, so these world poses
# are also planner-base poses.
CAP_GROCERY_OBJECT_POSE = (
    (0.3424806594848633, 0.14789724349975586, 0.05477798730134964),
    (0.0, 0.0, 0.0, 1.0),
)
CAP_GROCERY_BIN_POSE = (
    (0.4594290554523468, -0.15210512280464172, 0.01300068385899067),
    (0.0, 0.0, 0.0, 1.0),
)

# Local procedural support matching the successful Maple controlled-layout
# footprint without depending on the unpromoted Maple USD. Its top is at
# z=0.0030007, matching the recorded object/bin support plane.
CAP_GROCERY_SUPPORT_SIZE = (0.7, 1.0, 0.04)
CAP_GROCERY_SUPPORT_POSE = (
    (0.5485909, 0.0220630, -0.0169993),
    (0.0, 0.0, 0.0, 1.0),
)

# DROID home used by the successful grocery-packing profile: seven arm joints,
# the commanded finger joint, then five passive Robotiq linkage joints.
CAP_GROCERY_DROID_HOME = (
    0.0,
    -0.16104,
    0.0,
    -2.4446,
    0.0,
    2.22675,
    0.7854,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
