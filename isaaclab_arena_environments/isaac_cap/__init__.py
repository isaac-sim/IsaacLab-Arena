# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Cap environments and shared support code."""

__all__ = ["register_components"]


def register_components() -> None:
    """Register all available Isaac Cap components."""
    from .registration import register_components as _register_components

    _register_components()
