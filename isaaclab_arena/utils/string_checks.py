# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check strings that Arena records as single-line identifiers."""


def is_single_line_nonempty_string(value: object) -> bool:
    """Return whether a value is a non-empty string without line breaks."""
    return isinstance(value, str) and bool(value) and "\n" not in value and "\r" not in value
