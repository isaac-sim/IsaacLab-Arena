# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation system public API.

Names are exposed lazily (PEP 562 ``__getattr__``) so importing this package —
or its lightweight base / sampler submodules — never eagerly pulls heavy,
import-order-sensitive dependencies. See the comment on ``_EXPORTS`` for why.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> submodule that defines it. Resolved on first access.
#
# The laziness matters: some concrete variations (e.g. CameraDecalibrationVariation)
# import both torch and isaaclab.sensors, and importing that pair *before* the
# Isaac Sim SimulationApp launches corrupts the USD Python bindings. Eager imports
# here would propagate that hazard to anything that imports this package (or a
# submodule of it) at module-load time — e.g. pytest collecting a test file.
_EXPORTS = {
    "BuildTimeVariationBase": "variation_base",
    "RunTimeVariationBase": "variation_base",
    "VariationBase": "variation_base",
    "VariationBaseCfg": "variation_base",
    "SamplerBase": "sampler_base",
    "SamplerBaseCfg": "sampler_base",
    "CategoricalSampler": "categorical_sampler",
    "CategoricalSamplerCfg": "categorical_sampler",
    "UniformSampler": "uniform_sampler",
    "UniformSamplerCfg": "uniform_sampler",
    "HDRImageVariation": "hdr_image_variation",
    "HDRImageVariationCfg": "hdr_image_variation",
    "CameraDecalibrationVariation": "camera_decalibration",
    "CameraDecalibrationVariationCfg": "camera_decalibration",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:
    from isaaclab_arena.variations.camera_decalibration import (
        CameraDecalibrationVariation,
        CameraDecalibrationVariationCfg,
    )
    from isaaclab_arena.variations.categorical_sampler import CategoricalSampler, CategoricalSamplerCfg
    from isaaclab_arena.variations.hdr_image_variation import HDRImageVariation, HDRImageVariationCfg
    from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
    from isaaclab_arena.variations.variation_base import (
        BuildTimeVariationBase,
        RunTimeVariationBase,
        VariationBase,
        VariationBaseCfg,
    )


def __getattr__(name: str):
    """Lazily import and return a public variation name (see module docstring)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
