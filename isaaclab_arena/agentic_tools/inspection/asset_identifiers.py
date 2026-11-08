# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Identifiers shared by snapshot rendering and asset cards."""


def object_set_member_key(object_set_id: str, registry_name: str) -> str:
    """Return the thumbnail and AABB lookup key for one member of an object set."""
    return f"{object_set_id}::{registry_name}"
