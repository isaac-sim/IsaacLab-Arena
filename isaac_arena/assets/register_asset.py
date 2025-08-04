# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


from isaac_arena.assets.asset_registry import AssetRegistry


# Decorator to register an asset with the AssetRegistry.
def registerasset(cls):
    if AssetRegistry().is_registered(cls.name):
        print(f"WARNING: Asset {cls.name} is already registered. Doing nothing.")
    else:
        AssetRegistry().register(cls)
    return cls
