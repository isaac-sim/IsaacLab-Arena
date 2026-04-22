# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

##


# Option 1: Variations travel with asset.
# Decision: AGAINST

asset_registry = AssetRegistry()
apple = asset_registry.get_asset_by_name("apple")
apple.get_variation("color").enable()
apple.get_variation("color").set_sampler(UniformSampler(low=(0.0,) * 3, high=(1.0,) * 3))

# Option 2: Variations are added objects.
# Decision: SUPPORTED

asset_registry = AssetRegistry()
apple = asset_registry.get_asset_by_name("apple")
color_variation = ObjectColorVariation(apple, sampler=UniformSampler(low=(0.0,) * 3, high=(1.0,) * 3))
