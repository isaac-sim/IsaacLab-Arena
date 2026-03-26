# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
        assert name is not None, "Name is required for all assets"
        self.name = name
        self.tags = tags
