from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser("cuda_set_device_bench")
    parser.add_argument("--devices", type=str, default="0,1", help="Comma-separated logical CUDA device indices.")
    parser.add_argument("--same-device-iters", type=int, default=20000)
    parser.add_argument("--alternate-iters", type=int, default=20000)
    parser.add_argument("--alternate-touch-iters", type=int, default=2000)
    args = parser.parse_args()

    devices = [int(part.strip()) for part in args.devices.split(",") if part.strip()]
    if not devices:
        raise ValueError("At least one device index is required.")

    result: dict[str, object] = {
        "devices": devices,
        "cuda_visible_devices": None,
        "first_touch_ms": {},
    }

    try:
        import os

        result["cuda_visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
    except Exception:
        result["cuda_visible_devices"] = None

    # First touch per device: this includes CUDA context creation cost on this thread.
    for idx in devices:
        dev = f"cuda:{idx}"
        t0 = time.perf_counter()
        torch.cuda.set_device(idx)
        torch.empty(1, device=dev, dtype=torch.uint8)
        torch.cuda.synchronize(idx)
        result["first_touch_ms"][str(idx)] = (time.perf_counter() - t0) * 1000.0

    primary = devices[0]
    torch.cuda.set_device(primary)

    t0 = time.perf_counter()
    for _ in range(args.same_device_iters):
        torch.cuda.set_device(primary)
    result["same_device_avg_us"] = (time.perf_counter() - t0) * 1e6 / args.same_device_iters

    t0 = time.perf_counter()
    for i in range(args.alternate_iters):
        torch.cuda.set_device(devices[i % len(devices)])
    result["alternate_device_avg_us"] = (time.perf_counter() - t0) * 1e6 / args.alternate_iters

    t0 = time.perf_counter()
    for i in range(args.alternate_touch_iters):
        dev_idx = devices[i % len(devices)]
        dev = f"cuda:{dev_idx}"
        torch.cuda.set_device(dev_idx)
        torch.empty(1, device=dev, dtype=torch.uint8)
        torch.cuda.synchronize(dev_idx)
    result["alternate_with_touch_avg_ms"] = (time.perf_counter() - t0) * 1e3 / args.alternate_touch_iters

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
