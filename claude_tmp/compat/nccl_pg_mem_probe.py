from __future__ import annotations

import json
import os
import socket
import time

import torch
import torch.distributed as dist


def _snapshot(label: str) -> dict[str, object]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "label": label,
        "time_s": time.time(),
        "device": str(torch.cuda.current_device()),
        "free_bytes": int(free_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Create a CUDA context before measuring communicator overhead.
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)

    snapshots: list[dict[str, object]] = [_snapshot("after_cuda_context")]

    dist.init_process_group(backend="nccl", timeout=torch.distributed.constants.default_pg_timeout)
    dist.barrier()
    snapshots.append(_snapshot("after_main_pg_init"))

    work = torch.ones(1024, device=device, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(work)
    torch.cuda.synchronize(device)
    dist.barrier()
    snapshots.append(_snapshot("after_main_pg_all_reduce"))

    extra_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    dist.barrier()
    snapshots.append(_snapshot("after_extra_pg_init"))

    extra = torch.ones(1024, device=device, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(extra, group=extra_group)
    torch.cuda.synchronize(device)
    dist.barrier()
    snapshots.append(_snapshot("after_extra_pg_all_reduce"))

    report = {
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "snapshots": snapshots,
        "main_sum": float(work[0].item()),
        "extra_sum": float(extra[0].item()),
    }
    print(json.dumps(report, sort_keys=True), flush=True)

    # Clean shutdown: all ranks participate.
    dist.barrier()
    dist.destroy_process_group(extra_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
