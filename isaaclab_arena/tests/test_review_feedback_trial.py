# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Temporary sample for exercising review feedback."""


def test_environment_names_are_sorted():
    environment_names = ["kitchen", "warehouse", "tabletop"]

    sorted_environment_names = sorted(environment_names)

    assert sorted_environment_names == ["kitchen", "tabletop", "warehouse"]
