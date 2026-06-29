# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for reading Isaac Lab asset state buffers.

Some Isaac Lab link-frame quantities (e.g. ``root_link_pos_w``,
``body_link_pos_w``) are exposed as Warp arrays, which do not support
element indexing. :func:`to_torch` normalises a buffer to a torch tensor so the
rest of the pipeline can index/clone it uniformly, regardless of whether the
underlying buffer is already a torch tensor or a Warp array.

Imports of ``torch`` / ``warp`` are deferred to keep this module importable
without Isaac Sim.
"""

from __future__ import annotations

from typing import Any


def to_torch(array: Any) -> Any:
    """Return *array* as a torch tensor, converting from a Warp array if needed.

    Args:
        array: A torch tensor or a Warp array (as returned by Isaac Lab asset
            ``.data`` buffers).

    Returns:
        The same data as a torch tensor (zero-copy when possible). Inputs that
        are already torch tensors are returned unchanged.
    """
    import torch

    if isinstance(array, torch.Tensor):
        return array
    import warp as wp

    return wp.to_torch(array)
