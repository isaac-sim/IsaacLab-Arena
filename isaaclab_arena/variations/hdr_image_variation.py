# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that picks an HDR environment map for a dome light.

The HDR is sampled once before env-cfg composition and applied via
:meth:`~isaaclab_arena.assets.object_library.DomeLight.add_hdr`.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import DomeLight


@configclass
class HDRImageVariationCfg(VariationBaseCfg):
    """Configuration for :class:`HDRImageVariation`."""

    hdr_names: list[str] = field(default_factory=list)
    """Registered HDR names to sample from; empty means sample over every registered HDR."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform distribution over the resolved HDR pool."""


class HDRImageVariation(BuildTimeVariationBase):
    """Sample a single HDR and attach it to a :class:`DomeLight` at build time.

    Args:
        light: The dome light to mutate. A reference is captured; ``apply``
            mutates this instance.
        cfg: Tunable parameters. Defaults to sampling over every registered HDR.
            Override the choice sampler via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"hdr_image"``.
    """

    cfg: HDRImageVariationCfg

    def __init__(
        self,
        light: DomeLight,
        cfg: HDRImageVariationCfg | None = None,
        name: str = "hdr_image",
    ):
        super().__init__(cfg=cfg if cfg is not None else HDRImageVariationCfg(), name=name)
        self._light = light

    def apply(self) -> None:
        from isaaclab_arena.assets.hdr_image import HDRImage  # noqa: PLC0415
        from isaaclab_arena.assets.registries import HDRImageRegistry  # noqa: PLC0415

        registry = HDRImageRegistry()
        if self.cfg.hdr_names:
            for name in self.cfg.hdr_names:
                assert registry.is_registered(name), (
                    f"HDRImageVariation: HDR name '{name}' is not registered. "
                    f"Registered HDRs: {sorted(registry.get_all_keys())}."
                )
            hdr_names = self.cfg.hdr_names
        else:
            hdr_names = registry.get_all_keys()
            assert hdr_names, "HDRImageVariation: no HDRs are registered; cannot sample."

        assert self.sampler is not None, "HDRImageVariation: sampler not set."
        # Pass HDR names as the choice sampler's choices.
        hdr_name = self.sampler.sample(num_samples=1, choices=hdr_names)[0]
        hdr_cls: type[HDRImage] = registry.get_hdr_by_name(hdr_name)
        self._light.add_hdr(hdr_cls())
