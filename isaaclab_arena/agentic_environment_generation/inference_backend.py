# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the shared inference backend."""

from isaaclab_arena.inference.backend import (
    DEFAULT_ENDPOINT_NAME,
    INFERENCE_ENDPOINT_ENV_VAR,
    INFERENCE_ENDPOINTS,
    INTERNAL_ENDPOINT,
    MAX_RETRIES_LIMIT,
    OPENAI_ENDPOINT,
    PUBLIC_ENDPOINT,
    InferenceBackend,
    InferenceEndpoint,
    StructuredOutputRequest,
    build_strict_schema,
    resolve_inference_endpoint,
)

__all__ = [
    "DEFAULT_ENDPOINT_NAME",
    "INFERENCE_ENDPOINT_ENV_VAR",
    "INFERENCE_ENDPOINTS",
    "INTERNAL_ENDPOINT",
    "MAX_RETRIES_LIMIT",
    "OPENAI_ENDPOINT",
    "PUBLIC_ENDPOINT",
    "InferenceBackend",
    "InferenceEndpoint",
    "StructuredOutputRequest",
    "build_strict_schema",
    "resolve_inference_endpoint",
]
