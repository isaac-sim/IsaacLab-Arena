# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch


def setup_distributed() -> tuple[int, int]:
    """
    Initialize distributed when running under torchrun (WORLD_SIZE > 1).
    One process per GPU: each rank uses cuda:LOCAL_RANK (policy and AppLauncher pick this up).

    Returns:
        (rank, world_size). Single-process: (0, 1).
    """
    if "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        return 0, 1
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()
