# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Object-reference path conversion for standalone parent-asset USD stages."""

from __future__ import annotations

from typing import Any


def object_reference_relative_prim_path(ref, assets_by_node_id: dict[str, Any]) -> str:
    """Return an object reference's prim suffix beneath its parent asset."""
    reference_asset = assets_by_node_id[ref.id]
    parent_asset = assets_by_node_id[ref.parent_id]
    reference_path = reference_asset.get_prim_path().rstrip("/")
    parent_path = parent_asset.get_prim_path().rstrip("/")
    prefix = parent_path + "/"
    assert reference_path.startswith(prefix), (
        f"Object reference '{ref.id}' path '{reference_path}' is not beneath "
        f"parent '{ref.parent_id}' path '{parent_path}'"
    )
    return reference_path[len(prefix) :]
