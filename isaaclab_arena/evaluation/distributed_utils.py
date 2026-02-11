# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed sync without NCCL (file-based) so Isaac Sim GPU usage does not hang collectives."""

import os
import tempfile
import time


def file_barrier(rank: int, world_size: int, name: str = "arena_barrier", timeout_sec: float = 300.0) -> None:
    """
    Barrier using marker files so all ranks reach the same point. Safe to use inside
    SimulationApp (no NCCL); avoids hang when Isaac Sim holds the GPU.
    """
    if world_size <= 1:
        return
    tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
    # Same dir for all ranks in this run (torchrun sets MASTER_ADDR/MASTER_PORT); getppid() can differ per rank
    master = f"{os.environ.get('MASTER_ADDR', '')}_{os.environ.get('MASTER_PORT', '')}"
    barrier_dir = os.path.join(tmp, f"{name}_{master}")
    print(f"[Rank {rank}/{world_size}] Barrier directory: {barrier_dir}")
    os.makedirs(barrier_dir, exist_ok=True)
    marker = os.path.join(barrier_dir, f"rank_{rank}")
    with open(marker, "w") as f:
        f.flush()
        os.fsync(f.fileno())
    # Ensure directory listing is updated so other ranks see this file (e.g. on NFS)
    try:
        dfd = os.open(barrier_dir, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass
    start = time.monotonic()
    while True:
        n = sum(1 for i in range(world_size) if os.path.isfile(os.path.join(barrier_dir, f"rank_{i}")))
        if n >= world_size:
            break
        if time.monotonic() - start > timeout_sec:
            raise TimeoutError(f"file_barrier timed out after {timeout_sec}s (rank {rank}, saw {n}/{world_size})")
        time.sleep(0.01)
    try:
        for i in range(world_size):
            p = os.path.join(barrier_dir, f"rank_{i}")
            if os.path.isfile(p):
                os.remove(p)
    except OSError:
        pass
