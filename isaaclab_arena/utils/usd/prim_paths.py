# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Parse and resolve exact USD prim paths relative to an asset root."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Sdf, Usd


def parse_relative_prim_path(relative_path: str) -> Sdf.Path:
    """Parse a relative prim path, rejecting syntax that can escape its root.

    Args:
        relative_path: Exact relative path such as 'finger/collision', or '.' for the root.

    Returns:
        A relative Sdf.Path; no stage lookup is performed. Absolute paths, parent traversal,
        property paths, variant selections, and invalid prim-path syntax are rejected.
    """
    # Defer USD imports until SimulationApp has initialized.
    from pxr import Sdf

    assert isinstance(relative_path, str) and relative_path, "Prim path must be a nonempty relative string."
    path = Sdf.Path(relative_path)
    assert (
        not path.IsAbsolutePath()
        and (path.IsPrimPath() or path == Sdf.Path.reflexiveRelativePath)
        and ".." not in relative_path.split("/")
        and "{" not in relative_path
    ), f"Prim path must be relative to its root: {relative_path!r}"
    return path


def get_prim_relative_to_root(root: Usd.Prim, relative_path: str) -> Usd.Prim:
    """Return an existing prim selected by an exact path relative to root.

    Args:
        root: Prim anchoring the lookup on its stage.
        relative_path: Exact relative path such as 'finger/collision', or '.' for root itself.

    Returns:
        The resolved Usd.Prim. Instance proxies are allowed; callers that edit the prim
        must check editability themselves.
    """
    # Validate the path before resolving it on the root's stage.
    path = parse_relative_prim_path(relative_path)
    prim = root.GetStage().GetPrimAtPath(path.MakeAbsolutePath(root.GetPath()))
    assert prim.IsValid(), f"Prim does not exist: {root.GetPath()}/{relative_path}"
    return prim
