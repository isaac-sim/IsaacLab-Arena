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
  - install the CUDA-matched nvcomp wheel (e.g. ``nvidia-nvcomp-cu12`` for
    CUDA 12.x, ``nvidia-nvcomp-cu13`` for CUDA 13.x), not ``nvidia-nvcomp``
  - ``Codec(algorithm="LZ4")`` — keyword arg required
  - ``codec.decode()`` returns int8 dtype; use ``.view(torch.uint8)`` to fix
  - Isaac Sim containers need ``nvidia.__path__`` fix for the namespace package

Note on streams:
  nvcomp's Python ``Codec`` accepts an explicit ``cuda_stream`` parameter.
  If omitted, the docs say it creates an internal CUDA stream for the device.
  We therefore bind both the input/output DLPack wrappers and the ``Codec``
  itself to the active PyTorch stream on every call, rather than relying on
  post-hoc ``torch.cuda.current_stream().synchronize()`` guesses.
  See ``claude_tmp/test_nvcomp_sync.py`` for the empirical stream-behavior
  check that motivated this cleanup.
"""

from __future__ import annotations

import torch

# Module-level cache for nvcomp imports (populated on first use)
_nvcomp_from_dlpack = None
_nvcomp_import_attempted = False
_nvcomp_available = False

# Cached nvcomp symbols (populated on first use)
_nvcomp_Codec = None


def _ensure_nvcomp():
    """Import nvcomp and cache the result at module level.

    This function is a defensive guard — it should only be called on code
    paths that were reached AFTER negotiation confirmed nvcomp is available.
    The ``ImportError`` is therefore a programming error (misconfigured
    environment), not a normal fallback trigger.  Fallback to lz4 or no
    compression happens during capability negotiation, before this is called.
    """
    global _nvcomp_from_dlpack, _nvcomp_Codec, _nvcomp_import_attempted, _nvcomp_available

    if _nvcomp_import_attempted:
        if _nvcomp_available:
            return _nvcomp_from_dlpack
        raise ImportError(
            "A CUDA-matched nvidia-nvcomp wheel is required for GPU compression. "
            "Install the package matching your CUDA major version "
            "(for example `pip install nvidia-nvcomp-cu12` for CUDA 12.x "
            "or `pip install nvidia-nvcomp-cu13` for CUDA 13.x)."
        )

    _nvcomp_import_attempted = True

    try:
        from nvidia.nvcomp import Codec, from_dlpack
        _nvcomp_from_dlpack = from_dlpack
        _nvcomp_Codec = Codec
        _nvcomp_available = True
        return from_dlpack
    except ImportError:
        pass

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
        _nvcomp_from_dlpack = from_dlpack
        _nvcomp_Codec = Codec
        _nvcomp_available = True
        return from_dlpack
    except ImportError:
        raise ImportError(
            "A CUDA-matched nvidia-nvcomp wheel is required for GPU compression. "
            "Install the package matching your CUDA major version "
            "(for example `pip install nvidia-nvcomp-cu12` for CUDA 12.x "
            "or `pip install nvidia-nvcomp-cu13` for CUDA 13.x)."
        )


def _current_stream_info(tensor: torch.Tensor) -> tuple[int, int]:
    if not tensor.is_cuda:
        raise ValueError("GPU compression expects a CUDA tensor.")
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device=tensor.device)
    return device_index, int(stream.cuda_stream)


def _make_codec(device_index: int, cuda_stream: int):
    _ensure_nvcomp()
    if _nvcomp_Codec is None:
        raise ImportError("nvcomp Codec is unavailable after successful import setup.")
    # Do not cache codecs across arbitrary stream pointers. A Codec binds to a
    # specific cudaStream_t, and caller-managed non-default streams may be short-lived.
    return _nvcomp_Codec(algorithm="LZ4", device_id=device_index, cuda_stream=cuda_stream)


def gpu_compress(tensor: torch.Tensor) -> torch.Tensor:
    """Compress a contiguous GPU tensor using nvcomp LZ4.

    Args:
        tensor: A contiguous CUDA tensor (any dtype).

    Returns:
        A 1-D uint8 CUDA tensor containing the compressed data.
    """
    from_dlpack = _ensure_nvcomp()
    device_index, cuda_stream = _current_stream_info(tensor)
    codec = _make_codec(device_index, cuda_stream)

    # Flatten to 1-D uint8 and hand off the buffer on the active PyTorch stream.
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    nvcomp_input = from_dlpack(flat, cuda_stream=cuda_stream)
    compressed_nv = codec.encode(nvcomp_input)

    compressed_torch = torch.from_dlpack(compressed_nv.to_dlpack(cuda_stream=cuda_stream))
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
    from_dlpack = _ensure_nvcomp()
    device_index, cuda_stream = _current_stream_info(compressed)
    codec = _make_codec(device_index, cuda_stream)

    nvcomp_input = from_dlpack(compressed.contiguous(), cuda_stream=cuda_stream)
    decompressed_nv = codec.decode(nvcomp_input)

    result = torch.from_dlpack(decompressed_nv.to_dlpack(cuda_stream=cuda_stream))
    # nvcomp decode returns int8 — fix dtype
    if result.dtype == torch.int8:
        result = result.view(torch.uint8)
    return result[:original_nbytes]
