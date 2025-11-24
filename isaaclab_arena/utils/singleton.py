# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class SingletonMeta(type):
    """
    Metaclass that overrides __call__ so that only one instance
    of any class using it is ever created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # first time: actually create the instance
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # afterwards: always return the same object
        return cls._instances[cls]
