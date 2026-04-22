# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

##


# Option 1: Variations travel with asset.
# DECISION: SUPPORTED
# Reason: Certian variations will be specific to a single asset, for example
#         embodiment specific variations. Therefore it makes sense that variations
#         travel with the asset. The second thing is that then, for default variations
#         the user doesn't need to deal with them in the environment file, they are
#         automatically there (disabled by default).

asset_registry = AssetRegistry()
apple = asset_registry.get_asset_by_name("apple")
apple.get_variation("color").enable()
apple.get_variation("color").set_sampler(UniformSampler(low=(0.0,) * 3, high=(1.0,) * 3))

# Option 2: Variations are added objects.
# DECISION: AGAINST

asset_registry = AssetRegistry()
apple = asset_registry.get_asset_by_name("apple")
color_variation = ObjectColorVariation(apple, sampler=UniformSampler(low=(0.0,) * 3, high=(1.0,) * 3))
