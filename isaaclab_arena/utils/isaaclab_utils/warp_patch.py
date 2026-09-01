# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for converting Warp arrays to PyTorch tensors."""

from __future__ import annotations

from typing import Any

_EMPTY_CPU_TO_TORCH_WORKAROUND_INSTALLED = False


def install_empty_cpu_warp_to_torch_patch() -> None:
    """Use DLPack when the installed Warp cannot convert empty CPU arrays to Torch."""
    global _EMPTY_CPU_TO_TORCH_WORKAROUND_INSTALLED

    if _EMPTY_CPU_TO_TORCH_WORKAROUND_INSTALLED:
        return

    import torch

    import warp as wp

    original_to_torch = wp.to_torch
    probe = wp.empty((1, 0), dtype=wp.float32, device="cpu")
    try:
        original_to_torch(probe)
    except (TypeError, ValueError):
        pass
    else:
        _EMPTY_CPU_TO_TORCH_WORKAROUND_INSTALLED = True
        return

    def to_torch_with_empty_cpu_fallback(array: Any, requires_grad: bool | None = None) -> torch.Tensor:
        """Convert empty CPU arrays through DLPack and delegate all other arrays."""
        if isinstance(array, wp.array) and array.device.is_cpu and array.size == 0:
            if requires_grad is None:
                requires_grad = array.requires_grad

            tensor = torch.from_dlpack(wp.to_dlpack(array))
            tensor.requires_grad_(requires_grad)
            tensor._warp_array = array

            if requires_grad and array.grad is not None:
                tensor.grad = torch.from_dlpack(wp.to_dlpack(array.grad))
                tensor.grad._warp_grad_array = array.grad

            return tensor

        return original_to_torch(array, requires_grad=requires_grad)

    wp.to_torch = to_torch_with_empty_cpu_fallback
    _EMPTY_CPU_TO_TORCH_WORKAROUND_INSTALLED = True
