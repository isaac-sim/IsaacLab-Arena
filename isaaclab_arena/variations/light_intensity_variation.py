# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import LightBase


@configclass
class LightIntensityVariationCfg(VariationBaseCfg):
    """Configuration for LightIntensityVariation."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[100.0], high=[2000.0]))
    """Uniform distribution over light intensity."""


class LightIntensityVariation(BuildTimeVariationBase):
    """Sample a single intensity and apply it to a light at build time.

    Args:
        light: The light to mutate.
        cfg: Tunable parameters. LightIntensityVariationCfg
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: LightIntensityVariationCfg

    def __init__(
        self,
        light: LightBase,
        cfg: LightIntensityVariationCfg | None = None,
        name: str = "intensity",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightIntensityVariationCfg(), name=name)
        self._light = light

    def apply(self) -> None:
        assert self.sampler is not None, "LightIntensityVariation: sampler not set."
        intensity = float(self.sampler.sample(num_samples=1)[0, 0])
        self._light.set_intensity(intensity)
