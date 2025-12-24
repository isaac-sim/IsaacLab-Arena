# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict

import msgpack
import numpy as np


class MessageSerializer:
    """Msgpack-based serializer for dict-based policy messages.

    Supports:
    - standard Python types,
    - dataclasses (via to_json_serializable),
    - numpy.ndarray (tagged as __ndarray_class__),
    - generic binary blobs (tagged as __blob_class__).
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """Serialize a Python object to bytes using msgpack."""
        return msgpack.packb(data, default=MessageSerializer._encode_custom)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """Deserialize bytes into Python objects, decoding custom tags."""
        return msgpack.unpackb(data, object_hook=MessageSerializer._decode_custom)

    # ------------------------------------------------------------------ #
    # Custom encode / decode
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_custom(obj: Any) -> Any:
        """Decode tagged structures created in _encode_custom."""
        if not isinstance(obj, dict):
            return obj

        # numpy array
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)

        # generic binary blob
        if "__blob_class__" in obj:
            return {
                "mime": obj.get("mime"),
                "data": obj.get("as_bytes"),
            }

        # other tagged types can be added here
        return obj

    @staticmethod
    def _encode_custom(obj: Any) -> Any:
        """Encode special Python objects into msgpack-friendly structures."""

        # numpy array -> npy bytes
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}

        # generic binary blob: bytes / bytearray
        if isinstance(obj, (bytes, bytearray)):
            return {
                "__blob_class__": True,
                "mime": None,
                "as_bytes": bytes(obj),
            }

        # optional: custom Image/Frame types with to_bytes() and mime attribute
        if hasattr(obj, "to_bytes") and hasattr(obj, "mime"):
            return {
                "__blob_class__": True,
                "mime": getattr(obj, "mime"),
                "as_bytes": obj.to_bytes(),
            }

        # fall back to JSON-serializable representation
        return to_json_serializable(obj)


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    This is useful when encoding configuration objects or metadata.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # Fallback: convert to string
        return str(obj)

