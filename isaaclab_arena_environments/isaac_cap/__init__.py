# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Cap environments and shared validation tooling."""

_GEAR_INSERTION_EXPORTS = {
    "GearInsertionEasyNewtonEnvironment",
    "GearInsertionEasyNewtonEnvironmentCfg",
    "GearInsertionNewtonEnvironment",
    "GearInsertionNewtonEnvironmentCfg",
}

__all__ = sorted(_GEAR_INSERTION_EXPORTS)


def __getattr__(name: str):
    """Load gear-insertion exports without importing simulation modules eagerly."""
    if name in _GEAR_INSERTION_EXPORTS:
        from . import gear_insertion

        return getattr(gear_insertion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
