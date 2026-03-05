# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU-side tensor compression / decompression using nvcomp (LZ4).

Used in the UCX zero-copy path (Phase 6):
  - Client: compress observation tensors on GPU before UCX send
  - Server: decompress on GPU after UCX recv, before policy inference

The nvcomp codec operates via DLPack, so we go
  torch.Tensor → DLPack → nvcomp encode/decode → DLPack → torch.Tensor
without ever touching the CPU.

Important quirks (discovered during pre-test):
  - ``nvidia-nvcomp-cu12`` (NOT ``nvidia-nvcomp``) is the correct pip package
  - ``Codec(algorithm="LZ4")`` — keyword arg required
  - ``codec.decode()`` returns int8 dtype; use ``.view(torch.uint8)`` to fix
  - Isaac Sim containers need ``nvidia.__path__`` fix for the namespace package
"""

from __future__ import annotations

import torch


def _ensure_nvcomp():
    """Lazy import with Isaac Sim namespace fix."""
    try:
        from nvidia.nvcomp import Codec, from_dlpack
        return Codec, from_dlpack
    except ImportError:
        # Isaac Sim nvidia namespace conflict — try the path fix
        try:
            import site
            import nvidia
            sp = site.getsitepackages()
            for p in sp:
                nvidia_path = f"{p}/nvidia"
                if nvidia_path not in nvidia.__path__:
                    nvidia.__path__.append(nvidia_path)
            from nvidia.nvcomp import Codec, from_dlpack
            return Codec, from_dlpack
        except ImportError:
            raise ImportError(
                "nvidia-nvcomp-cu12 is required for GPU compression. "
                "Install with: pip install nvidia-nvcomp-cu12"
            )


# Module-level codec singleton (created on first use)
_codec = None


def _get_codec():
    global _codec
    if _codec is None:
        Codec, _ = _ensure_nvcomp()
        _codec = Codec(algorithm="LZ4")
    return _codec


def gpu_compress(tensor: torch.Tensor) -> torch.Tensor:
    """Compress a contiguous GPU tensor using nvcomp LZ4.

    Args:
        tensor: A contiguous CUDA tensor (any dtype).

    Returns:
        A 1-D uint8 CUDA tensor containing the compressed data.
    """
    _, from_dlpack = _ensure_nvcomp()
    codec = _get_codec()

    # Flatten to 1-D uint8 view for nvcomp
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    nvcomp_input = from_dlpack(flat)
    compressed_nv = codec.encode(nvcomp_input)

    # Convert back to torch
    compressed_torch = torch.from_dlpack(compressed_nv)
    # nvcomp may return int8 — normalize to uint8
    if compressed_torch.dtype == torch.int8:
        compressed_torch = compressed_torch.view(torch.uint8)
    return compressed_torch


def gpu_decompress(compressed: torch.Tensor, original_nbytes: int) -> torch.Tensor:
    """Decompress a nvcomp-LZ4 compressed GPU tensor.

    Args:
        compressed: A 1-D uint8 CUDA tensor (output of ``gpu_compress``).
        original_nbytes: The original uncompressed size in bytes, needed to
            interpret the result correctly.

    Returns:
        A 1-D uint8 CUDA tensor of length ``original_nbytes``.
    """
    _, from_dlpack = _ensure_nvcomp()
    codec = _get_codec()

    nvcomp_input = from_dlpack(compressed.contiguous())
    decompressed_nv = codec.decode(nvcomp_input)

    result = torch.from_dlpack(decompressed_nv)
    # nvcomp decode returns int8 — fix dtype
    if result.dtype == torch.int8:
        result = result.view(torch.uint8)
    return result[:original_nbytes]
