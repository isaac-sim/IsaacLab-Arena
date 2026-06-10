# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that samples the intensity of a dome light.

The intensity is sampled once before env-cfg composition and applied via
:meth:`~isaaclab_arena.assets.object_library.DomeLight.set_intensity`.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import DomeLight


@configclass
class LightIntensityVariationCfg(VariationBaseCfg):
    """Configuration for :class:`LightIntensityVariation`."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[100.0], high=[2000.0]))
    """Uniform distribution over dome light intensity."""


class LightIntensityVariation(BuildTimeVariationBase):
    """Sample a single intensity and apply it to a :class:`DomeLight` at build time.

    Args:
        light: The dome light to mutate. A reference is captured; ``apply``
            mutates this instance.
        cfg: Tunable parameters. Defaults to sampling intensity uniformly over
            ``[1000.0, 5000.0]``. Override the distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"dome_light_intensity"``.
    """

    cfg: LightIntensityVariationCfg

    def __init__(
        self,
        light: DomeLight,
        cfg: LightIntensityVariationCfg | None = None,
        name: str = "intensity",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightIntensityVariationCfg(), name=name)
        self._light = light

    def apply(self) -> None:
        assert self.sampler is not None, "LightIntensityVariation: sampler not set."
        intensity = float(self.sampler.sample(num_samples=1)[0, 0])
        print(f"intensity: {intensity}")
        self._light.set_intensity(intensity)
