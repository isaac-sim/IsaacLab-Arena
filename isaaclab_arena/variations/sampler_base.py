# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC

from isaaclab.utils import configclass


@configclass
class SamplerBaseCfg:
    """Marker configclass for any sampler cfg.

    Concrete subclasses override :meth:`build` to return their live sampler.
    """

    def build(self) -> SamplerBase:
        """Return the live :class:`SamplerBase` described by this cfg."""
        raise NotImplementedError(
            f"{type(self).__name__}.build() is not implemented; every concrete SamplerBaseCfg "
            "subclass must provide a build() that returns its live sampler."
        )


class SamplerBase(ABC):
    """Marker base class shared by every sampler family."""
