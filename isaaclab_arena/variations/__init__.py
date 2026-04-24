# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation system public API.

Importing this package triggers registration of all built-in variations in
:class:`~isaaclab_arena.variations.base.VariationRegistry`.
"""

from isaaclab_arena.variations.object_color import ObjectColorVariation, ObjectColorVariationCfg
from isaaclab_arena.variations.sampler import Sampler, SamplerCfg, UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg
from isaaclab_arena.variations.variation_registry import VariationRegistry, register_variation

__all__ = [
    "ObjectColorVariation",
    "ObjectColorVariationCfg",
    "Sampler",
    "SamplerCfg",
    "UniformSampler",
    "UniformSamplerCfg",
    "VariationBase",
    "VariationBaseCfg",
    "VariationRegistry",
    "register_variation",
]
