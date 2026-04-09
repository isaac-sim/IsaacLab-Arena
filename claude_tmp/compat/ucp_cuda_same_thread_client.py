from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback

import ucp
import torch


def _prepare_cuda(device: str) -> torch.device:
    torch.cuda.set_device(device)
    dev = torch.device(device)
    # Force CUDA context creation on this thread.
    _ = torch.empty(1, device=dev, dtype=torch.uint8)
    torch.cuda.synchronize(dev)
    return dev


async def _run_client(host: str, port: int, nbytes: int, device: str) -> dict[str, object]:
    dev = _prepare_cuda(device)
    ep = await ucp.create_endpoint(host, port)

    send_buf = torch.arange(nbytes, dtype=torch.uint8, device=dev)
    recv_buf = torch.empty_like(send_buf)

    await ep.send(send_buf)
    await ep.recv(recv_buf)
    torch.cuda.synchronize(dev)
    await ep.close()

    return {
        "status": "ok",
        "device": str(dev),
        "nbytes": nbytes,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "send_sum": int(send_buf.sum().item()),
        "recv_sum": int(recv_buf.sum().item()),
        "equal": bool(torch.equal(send_buf, recv_buf)),
    }


def main() -> None:
    parser = argparse.ArgumentParser("ucp_cuda_same_thread_client")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=5580)
    parser.add_argument("--nbytes", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fast-exit", action="store_true")
    args = parser.parse_args()

    try:
        result = asyncio.run(_run_client(args.host, args.port, args.nbytes, args.device))
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.fast_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
