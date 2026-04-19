# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def _nvtx_push(name: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:
        return False
    return True


def _nvtx_pop(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch

        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    enabled = _nvtx_push(name)
    try:
        yield
    finally:
        _nvtx_pop(enabled)
