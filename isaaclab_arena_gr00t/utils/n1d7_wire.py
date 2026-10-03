# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode N1.7 responses left as envelopes by the pinned N1.6 client."""

import numpy as np
from typing import Any

NUMPY_ARRAY_MARKER = b"nd"


def decode_n1d7_response(value: Any) -> Any:
    """Decode numeric msgpack-numpy arrays and N1.7 modality metadata recursively.

    N1.7 accepts the N1.6 client's requests, but replies with msgpack-numpy
    arrays and a renamed ModalityConfig marker. Keep this compatibility at
    the response boundary without modifying the shared N1.6 serializer.
    """
    if isinstance(value, dict):
        if "__ModalityConfig__" in value:
            from gr00t.data.types import ModalityConfig

            return ModalityConfig(**value["as_json"])
        if NUMPY_ARRAY_MARKER in value:
            # GR00T sends numeric tensors. Never load object/pickle payloads.
            dtype = np.dtype(value[b"type"])
            if dtype.kind not in "biufc":
                raise ValueError(f"Unsupported GR00T response dtype: {dtype}")
            array = np.frombuffer(value[b"data"], dtype=dtype).copy()
            return array.reshape(value[b"shape"]) if value[NUMPY_ARRAY_MARKER] else array[0]
        return {key: decode_n1d7_response(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(decode_n1d7_response(item) for item in value)
    return value
