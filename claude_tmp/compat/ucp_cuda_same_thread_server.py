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


async def _run_server(port: int, nbytes: int, device: str) -> dict[str, object]:
    dev = _prepare_cuda(device)
    done = asyncio.Event()
    result: dict[str, object] = {}

    async def _on_connect(ep) -> None:
        recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        await ep.recv(recv_buf)
        torch.cuda.synchronize(dev)

        echoed = recv_buf.clone()
        await ep.send(echoed)
        await ep.close()

        result.update(
            {
                "status": "ok",
                "received_sum": int(recv_buf.sum().item()),
                "device": str(dev),
                "nbytes": nbytes,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            }
        )
        done.set()

    listener = ucp.create_listener(_on_connect, port=port)
    print(json.dumps({"status": "listening", "port": listener.port, "device": str(dev)}), flush=True)
    await done.wait()
    return result


def main() -> None:
    parser = argparse.ArgumentParser("ucp_cuda_same_thread_server")
    parser.add_argument("--port", type=int, default=5580)
    parser.add_argument("--nbytes", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fast-exit", action="store_true")
    args = parser.parse_args()

    try:
        result = asyncio.run(_run_server(args.port, args.nbytes, args.device))
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
