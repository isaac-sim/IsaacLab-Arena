# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


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
