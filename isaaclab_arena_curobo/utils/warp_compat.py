# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warp compatibility shim for cuRobo."""

from __future__ import annotations

import sys
import types

import warp as wp

# Torch interop functions that warp-lang 1.13 moved from ``warp.torch`` to the top-level ``warp`` namespace.
_WARP_TORCH_FUNCTION_NAMES = (
    "from_torch",
    "to_torch",
    "device_from_torch",
    "device_to_torch",
    "dtype_from_torch",
    "dtype_to_torch",
    "stream_from_torch",
    "stream_to_torch",
)


def add_warp_torch_namespace() -> None:
    """Re-expose ``warp.torch`` as a module forwarding to the top-level ``warp`` torch interop functions.

    warp-lang 1.13 dropped the ``warp.torch`` submodule, but cuRobo still calls ``wp.torch.device_from_torch``
    when it builds its mesh collision checker. Call this before constructing a cuRobo solver. Idempotent, and a
    no-op when ``warp.torch`` already exists. Mirrors Isaac Lab Mimic's shim; remove once the cuRobo pin uses
    the top-level ``wp.*`` API.
    """
    if hasattr(wp, "torch"):
        return
    warp_torch = types.ModuleType("warp.torch")
    for name in _WARP_TORCH_FUNCTION_NAMES:
        if hasattr(wp, name):
            setattr(warp_torch, name, getattr(wp, name))
    wp.torch = warp_torch
    sys.modules["warp.torch"] = warp_torch
