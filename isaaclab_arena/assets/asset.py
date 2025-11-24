# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class Asset:
    """
    Base class for all assets.
    """

    def __init__(self, name: str, tags: list[str] | None = None, **kwargs):
        # NOTE: Cooperative Multiple Inheritance Pattern.
        # Calling super even though this is a base class to support
        # multiple inheritance of inheriting classes.
        super().__init__(**kwargs)
        # self.name = name
        self._name = name
        self.tags = tags

    # name is a read-only property
    @property
    def name(self) -> str:
        return self._name
