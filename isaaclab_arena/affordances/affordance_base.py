# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC


class AffordanceBase(ABC):
    """Base class for affordances.

    NOTE: Affordances must always be combined with an Asset class through multiple inheritance.
    This ensures that affordances have access to the asset's name and other properties.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # NOTE(alexmillane, 2025.09.19) Affordances always have be combined with
        # an Asset which has a "name" property. We enforce this at runtime.
        from isaaclab_arena.assets.asset import Asset

        if not isinstance(self, Asset):
            raise TypeError(
                f"{self.__class__.__name__} must inherit from Asset. "
                "Affordances must be combined with Asset through multiple inheritance. "
                f"Example: class MyClass(Asset, {self.__class__.__bases__[0].__name__})."
            )
