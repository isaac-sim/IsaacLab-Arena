# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scalar range normalization helpers."""

import torch


def normalize_value(value: torch.Tensor, min_value: float, max_value: float):
    """Map a value from [min_value, max_value] to [0, 1].

    Args:
        value: Value to normalize.
        min_value: Lower bound of the range.
        max_value: Upper bound of the range.
    """
    return (value - min_value) / (max_value - min_value)


def unnormalize_value(value: float, min_value: float, max_value: float):
    """Map a value from [0, 1] to [min_value, max_value].

    Args:
        value: Normalized value in [0, 1].
        min_value: Lower bound of the target range.
        max_value: Upper bound of the target range.
    """
    return min_value + (max_value - min_value) * value
