# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that samples a continuous orientation for a distant light.

The orientation is parameterized by a 2D (azimuth, elevation).
"""

from __future__ import annotations

import math
import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_angle_axis

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_library import DirectionalLight, DomeLight


def quat_xyzw_from_azimuth_elevation(azimuth_rad: float, elevation_rad: float) -> tuple[float, float, float, float]:
    """Return an xyzw quaternion orienting a distant light to arrive from ``(azimuth, elevation)``.

    A USD DistantLight emits along its local -Z axis, so the returned quaternion rotates
    local +Z onto the incoming-light direction. elevation is measured down from directly
    above: elevation = 0 is top-down.

    Args:
        azimuth_rad: Angle around +Z axis [rad].
        elevation_rad: Angle down from directly above [rad]. 0 is directly overhead.

    Returns:
        The orientation quaternion as (x, y, z, w).
    """
    axis = torch.tensor([[-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0]])
    angle = torch.tensor([elevation_rad])
    return tuple(quat_from_angle_axis(angle, axis).reshape(-1).tolist())


@configclass
class LightDirectionVariationCfg(VariationBaseCfg):
    """Configuration for LightDirectionVariation.

    The sampler draws azimuth and elevation angles (in radians).
    """

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-math.pi, math.radians(0.0)],
            high=[math.pi, math.radians(80.0)],
        )
    )
    """Uniform distribution over (azimuth, elevation).

    The default range goes down to 80 degrees elevation to ensure we still cast shadows.
    """

    dome_intensity_when_active: float = 500.0
    """Intensity the registered dome light (if any) is dimmed to while this variation is active.

    Dimming the dome lets the directional light's shadows show through. Only applied when a dome
    light is registered via ``set_dome_light``.
    """


class LightDirectionVariation(BuildTimeVariationBase):
    """Sample a continuous orientation and apply it to a DirectionalLight.

    Args:
        light: The directional light to mutate.
        cfg: Parameters of type LightDirectionVariationCfg.
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: LightDirectionVariationCfg

    def __init__(
        self,
        light: DirectionalLight,
        cfg: LightDirectionVariationCfg | None = None,
        name: str = "direction",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightDirectionVariationCfg(), name=name)
        self._light = light
        self._dome_light: DomeLight | None = None

    def set_dome_light(self, dome_light: DomeLight) -> None:
        """Register a dome light to dim while this variation is active, so shadows are visible."""
        self._dome_light = dome_light

    def _prepare_at_build_time(self) -> None:
        """Prepare the scene for this variation, if activated.

        Preparation:
        - turns on directional light
        - (optional) dims a dome light, if registered with this variation.
        """
        self._light.on()
        if self._dome_light is not None:
            print(
                f"Warning: Dimming dome light to intensity: {self.cfg.dome_intensity_when_active} because the light"
                " direction variation is active.This can cause issues if combined with other variations that modify"
                " the dome light."
            )
            self._dome_light.set_intensity(self.cfg.dome_intensity_when_active)

    def _realize_at_build_time(self) -> None:
        assert self.sampler is not None, "LightDirectionVariation: sampler not set."
        azimuth, elevation = self.sampler.sample(num_samples=1)[0].tolist()
        self._light.set_orientation(quat_xyzw_from_azimuth_elevation(azimuth, elevation))
