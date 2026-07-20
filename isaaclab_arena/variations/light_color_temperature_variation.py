# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that samples a white-point color temperature for a light."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import LightBase


@configclass
class LightColorTemperatureVariationCfg(VariationBaseCfg):
    """Configuration for LightColorTemperatureVariation."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[1000.0], high=[10000.0]))
    """Uniform distribution over color temperature in Kelvin, from warm (low) to cool (high)."""


class LightColorTemperatureVariation(BuildTimeVariationBase):
    """Sample a color temperature and apply it to a light at build time.

    Args:
        light: The light to mutate.
        cfg: Tunable parameters. LightColorTemperatureVariationCfg
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: LightColorTemperatureVariationCfg

    def __init__(
        self,
        light: LightBase,
        cfg: LightColorTemperatureVariationCfg | None = None,
        name: str = "color_temperature",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightColorTemperatureVariationCfg(), name=name)
        self._light = light

    def apply(self) -> None:
        assert self.sampler is not None, "LightColorTemperatureVariation: sampler not set."
        color_temperature = float(self.sampler.sample(num_samples=1)[0, 0])
        self._light.set_color_temperature(color_temperature)
