# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared validation helpers for Evaluation protocol value objects."""


def validate_name(value: object, field_name: str) -> None:
    """Validate a non-empty string identifier without surrounding whitespace.

    Args:
        value: Candidate identifier.
        field_name: Public field name to include in validation errors.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty string without surrounding whitespace")
