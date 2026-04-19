# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor packing for remote-policy observation payloads.

Packs multiple observation tensors into a single flat uint8 buffer plus a
tensor_layout that records each tensor's shape/dtype/offset so the server can
reconstruct the original tensor dict.

Packing is required by both transport paths (inline ZMQ and dedicated mooncake
tensor channel). Compression is not applied here; the flat buffer is sent
as-is over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import torch

from isaaclab_arena.remote_policy.profiling import nvtx_range


@dataclass
class PackedTensorPayload:
    tensor_layout: list[dict[str, Any]]
    flat_buffer: "torch.Tensor"
    original_nbytes: int


def split_observation_entries(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, "torch.Tensor"] | None]:
    """Split observation entries without committing to a backend format."""
    control_entries: dict[str, Any] = {}
    tensor_entries: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            tensor_entries[key] = value.detach()
            continue
        control_entries[key] = value

    return control_entries, (tensor_entries or None)


def build_control_observation_for_tensor_transport(
    control_entries: dict[str, Any],
    tensor_entries: dict[str, "torch.Tensor"] | None,
) -> tuple[dict[str, Any], dict[str, "torch.Tensor"] | None]:
    """Build the control-plane observation for dedicated tensor transports.

    CUDA tensors stay on the dedicated tensor path. Non-CUDA tensors are
    materialized onto the legacy control observation dict.
    """
    control_observation = dict(control_entries)
    transport_tensor_entries: dict[str, torch.Tensor] = {}
    for key, tensor in (tensor_entries or {}).items():
        if tensor.is_cuda:
            transport_tensor_entries[key] = tensor
            continue
        control_observation[key] = tensor.cpu().numpy()
    return control_observation, (transport_tensor_entries or None)


class TensorPayloadCodec:
    """Pack multiple tensors into one flat buffer + layout metadata."""

    @staticmethod
    def _tensor_metadata(tensor: "torch.Tensor") -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
            "nbytes": int(tensor.numel() * tensor.element_size()),
        }

    @staticmethod
    def _debug_tensor_codec_enabled() -> bool:
        return os.getenv("ISAACLAB_ARENA_REMOTE_DEBUG_TENSOR_CODEC") == "1"

    @classmethod
    def _emit_tensor_codec_debug(cls, event: str, **payload: Any) -> None:
        if not cls._debug_tensor_codec_enabled():
            return
        debug_record = {
            "event": event,
            "case": os.getenv("ISAACLAB_ARENA_REMOTE_METRICS_CASE"),
            "role": os.getenv("ISAACLAB_ARENA_REMOTE_METRICS_ROLE"),
        }
        debug_record.update(payload)
        print(json.dumps(debug_record, ensure_ascii=False, sort_keys=True), flush=True)

    @staticmethod
    def _pack_tensor_entries(
        tensor_entries: dict[str, "torch.Tensor"],
        *,
        target_device: str,
    ) -> PackedTensorPayload:
        with nvtx_range("tensor.pack_entries"):
            tensor_layout: list[dict[str, Any]] = []
            flat_parts = []
            tensor_debug_entries: list[dict[str, Any]] = []
            offset = 0
            for key, tensor in tensor_entries.items():
                source_tensor = tensor
                if str(tensor.device) != target_device:
                    tensor = tensor.to(device=target_device)
                flat = tensor.contiguous().view(torch.uint8).reshape(-1)
                tensor_layout.append({
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "offset": offset,
                    "nbytes": flat.numel(),
                })
                flat_parts.append(flat)
                tensor_debug_entries.append(
                    {
                        "key": key,
                        "source_tensor": TensorPayloadCodec._tensor_metadata(source_tensor),
                        "packed_tensor": TensorPayloadCodec._tensor_metadata(tensor),
                        "flat_tensor": TensorPayloadCodec._tensor_metadata(flat),
                    }
                )
                offset += flat.numel()

            flat_buffer = torch.cat(flat_parts) if len(flat_parts) > 1 else flat_parts[0]
            original_nbytes = flat_buffer.numel()

        TensorPayloadCodec._emit_tensor_codec_debug(
            "tensor_payload_packed",
            target_device=target_device,
            tensor_count=len(tensor_debug_entries),
            tensors=tensor_debug_entries,
            flat_buffer=TensorPayloadCodec._tensor_metadata(flat_buffer),
            original_nbytes=original_nbytes,
        )
        return PackedTensorPayload(
            tensor_layout=tensor_layout,
            flat_buffer=flat_buffer,
            original_nbytes=original_nbytes,
        )

    def prepare_tensor_payload(
        self,
        tensor_entries: dict[str, "torch.Tensor"],
        *,
        target_device: str,
    ) -> PackedTensorPayload:
        return self._pack_tensor_entries(
            tensor_entries,
            target_device=target_device,
        )
