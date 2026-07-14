# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that samples an RGB color for a light."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import LightBase


@configclass
class LightColorVariationCfg(VariationBaseCfg):
    """Configuration for LightColorVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[1.0, 1.0, 1.0])
    )
    """Uniform distribution over the (red, green, blue) channels, each in [0, 1]."""


class LightColorVariation(BuildTimeVariationBase):
    """Sample an RGB color and apply it to a light at build time.

    Args:
        light: The light to mutate.
        cfg: Tunable parameters. LightColorVariationCfg
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: LightColorVariationCfg

    def __init__(
        self,
        light: LightBase,
        cfg: LightColorVariationCfg | None = None,
        name: str = "color",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightColorVariationCfg(), name=name)
        self._light = light

    def apply(self) -> None:
        assert self.sampler is not None, "LightColorVariation: sampler not set."
        color = tuple(self.sampler.sample(num_samples=1)[0].tolist())
        self._light.set_color(color)
