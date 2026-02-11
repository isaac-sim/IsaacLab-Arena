# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal distributed helpers for sync-only use (barrier / destroy). No init for sim device — use env vars + AppLauncher."""

import os

import torch


def setup_process_group_for_sync() -> tuple[int, int, int]:
    """
    Init process group only for barrier/destroy so all ranks exit together.
    Set CUDA device before init so NCCL uses the correct GPU.
    Call only when WORLD_SIZE > 1.

    Returns:
        (rank, world_size, local_rank).
    """
    if "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        return 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    return (
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        local_rank,
    )


def barrier(device_id: int | None = None) -> None:
    """Sync all processes. Set device_id so NCCL uses the correct GPU."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    torch.distributed.barrier()


def destroy_process_group(device_id: int | None = None) -> None:
    """Destroy process group so all ranks exit cleanly. Set device_id before destroy."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    torch.distributed.destroy_process_group()
