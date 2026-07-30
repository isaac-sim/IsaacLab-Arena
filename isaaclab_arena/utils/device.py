# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def resolve_cuda_device(device: str | torch.device | None) -> torch.device:
    """Pick an explicit CUDA device, defaulting to the current one."""
    if device is not None:
        return torch.device(device)
    assert torch.cuda.is_available(), "Requires a CUDA GPU."
    return torch.device(f"cuda:{torch.cuda.current_device()}")
