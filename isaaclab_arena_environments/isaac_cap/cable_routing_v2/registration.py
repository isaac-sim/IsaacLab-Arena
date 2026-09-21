# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for Isaac CAP component registration."""

from __future__ import annotations


def register_components() -> None:
    """Register all Isaac CAP components through their decorators."""
    from .. import register_components as register_all_components

    register_all_components()


__all__ = ["register_components"]
