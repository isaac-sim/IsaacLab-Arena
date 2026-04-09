from __future__ import annotations

import argparse
import importlib
import json
import os
import site
import sys
import traceback


def _apply_nvidia_namespace_fix() -> None:
    try:
        import nvidia
    except ImportError:
        return

    for p in site.getsitepackages():
        candidate = os.path.join(p, "nvidia")
        if os.path.isdir(candidate) and candidate not in nvidia.__path__:
            nvidia.__path__.append(candidate)


def _import_module(name: str) -> str:
    if name == "nvcomp":
        from nvidia.nvcomp import Codec  # noqa: F401

        return "nvidia.nvcomp"
    if name == "zmq":
        import zmq  # noqa: F401

        return "zmq"
    if name == "torch":
        import torch  # noqa: F401

        return "torch"
    if name == "ucp":
        import ucp  # noqa: F401

        return "ucp"
    importlib.import_module(name)
    return name


def main() -> None:
    parser = argparse.ArgumentParser("import_order_probe")
    parser.add_argument("--modules", type=str, required=True, help="Comma-separated import order, e.g. torch,ucp,nvcomp")
    parser.add_argument("--fix-nvidia-namespace", action="store_true")
    parser.add_argument("--touch-cuda", action="store_true")
    args = parser.parse_args()

    if args.fix_nvidia_namespace:
        _apply_nvidia_namespace_fix()

    imported: list[str] = []
    try:
        for raw_name in args.modules.split(","):
            name = raw_name.strip()
            if not name:
                continue
            imported.append(_import_module(name))

        if args.touch_cuda:
            import torch

            cuda_state = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
            if torch.cuda.is_available():
                cuda_state["current_device"] = torch.cuda.current_device()
                cuda_state["tensor_device"] = str(torch.zeros(1, device="cuda:0").device)
        else:
            cuda_state = None

        print(
            json.dumps(
                {
                    "status": "ok",
                    "imported": imported,
                    "fix_nvidia_namespace": args.fix_nvidia_namespace,
                    "touch_cuda": args.touch_cuda,
                    "cuda_state": cuda_state,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "imported": imported,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
