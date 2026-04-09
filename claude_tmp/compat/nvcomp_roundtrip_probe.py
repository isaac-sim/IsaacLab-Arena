from __future__ import annotations

import argparse
import json
import os
import site
import sys
import traceback

import torch


def _apply_nvidia_namespace_fix() -> None:
    try:
        import nvidia
    except ImportError:
        return

    for p in site.getsitepackages():
        candidate = os.path.join(p, "nvidia")
        if os.path.isdir(candidate) and candidate not in nvidia.__path__:
            nvidia.__path__.append(candidate)


def main() -> None:
    parser = argparse.ArgumentParser("nvcomp_roundtrip_probe")
    parser.add_argument("--fix-nvidia-namespace", action="store_true")
    parser.add_argument("--clone-compressed", action="store_true")
    parser.add_argument("--separate-codecs", action="store_true")
    parser.add_argument("--use-repo-helper", action="store_true")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.fix_nvidia_namespace:
        _apply_nvidia_namespace_fix()

    x = torch.arange(args.size, dtype=torch.uint8, device=args.device)

    try:
        if args.use_repo_helper:
            from isaaclab_arena.remote_policy.gpu_compression import gpu_compress, gpu_decompress

            compressed = gpu_compress(x)
            if args.clone_compressed:
                compressed = compressed.clone()
            result = gpu_decompress(compressed, x.numel())
        else:
            from nvidia.nvcomp import Codec, from_dlpack

            cuda_stream = int(torch.cuda.current_stream(device=x.device).cuda_stream)
            encode_codec = Codec(algorithm="LZ4", device_id=x.device.index or 0, cuda_stream=cuda_stream)
            decode_codec = (
                Codec(algorithm="LZ4", device_id=x.device.index or 0, cuda_stream=cuda_stream)
                if args.separate_codecs
                else encode_codec
            )

            flat = x.contiguous().view(torch.uint8).reshape(-1)
            nvcomp_input = from_dlpack(flat, cuda_stream=cuda_stream)
            compressed_nv = encode_codec.encode(nvcomp_input)
            compressed = torch.from_dlpack(compressed_nv.to_dlpack(cuda_stream=cuda_stream))
            if compressed.dtype == torch.int8:
                compressed = compressed.view(torch.uint8)
            if args.clone_compressed:
                compressed = compressed.clone()

            decode_input = from_dlpack(compressed.contiguous(), cuda_stream=cuda_stream)
            decompressed_nv = decode_codec.decode(decode_input)
            result = torch.from_dlpack(decompressed_nv.to_dlpack(cuda_stream=cuda_stream))
            if result.dtype == torch.int8:
                result = result.view(torch.uint8)

        compressed_meta = {
            "is_cuda": compressed.is_cuda,
            "device": str(compressed.device),
            "dtype": str(compressed.dtype),
            "shape": list(compressed.shape),
        }
        ok = bool(torch.equal(x, result[: x.numel()]))

        print(
            json.dumps(
                {
                    "status": "ok",
                    "compressed": compressed_meta,
                    "roundtrip_ok": ok,
                    "clone_compressed": args.clone_compressed,
                    "separate_codecs": args.separate_codecs,
                    "use_repo_helper": args.use_repo_helper,
                    "fix_nvidia_namespace": args.fix_nvidia_namespace,
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
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "clone_compressed": args.clone_compressed,
                    "separate_codecs": args.separate_codecs,
                    "use_repo_helper": args.use_repo_helper,
                    "fix_nvidia_namespace": args.fix_nvidia_namespace,
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
