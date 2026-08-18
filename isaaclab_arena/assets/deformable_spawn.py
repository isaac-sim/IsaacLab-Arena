# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral volume-deformable material and Newton spawn mapping."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg


@dataclass(frozen=True)
class VolumeDeformableMaterial:
    """Backend-neutral physical properties for a volume deformable."""

    youngs_modulus: float
    poissons_ratio: float
    density: float
    damping: float = 0.0
    particle_radius: float = 0.008

    def __post_init__(self) -> None:
        assert self.youngs_modulus > 0.0, "youngs_modulus must be positive"
        assert -1.0 < self.poissons_ratio < 0.5, "poissons_ratio must be in (-1, 0.5)"
        assert self.density > 0.0, "density must be positive"
        assert self.damping >= 0.0, "damping must be non-negative"
        assert self.particle_radius > 0.0, "particle_radius must be positive"


def lame_parameters(youngs_modulus: float, poissons_ratio: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
    k_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    k_lambda = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    return k_mu, k_lambda


def build_newton_volume_spawn(
    source: str | DeformableObjectSpawnerCfg,
    material: VolumeDeformableMaterial,
    *,
    visual_material: VisualMaterialCfg | None = None,
    scale: tuple[float, float, float] | None = None,
) -> DeformableObjectSpawnerCfg:
    """Map a volume source and neutral material to a Newton deformable spawner."""
    if isinstance(source, str):
        spawn_cfg = UsdFileCfg(usd_path=source, scale=scale)
    else:
        assert scale is None, "scale is only supported for USD deformable sources"
        spawn_cfg = copy.deepcopy(source)

    k_mu, k_lambda = lame_parameters(material.youngs_modulus, material.poissons_ratio)
    spawn_cfg.deformable_props = NewtonDeformableBodyPropertiesCfg()
    spawn_cfg.physics_material = NewtonDeformableBodyMaterialCfg(
        density=material.density,
        particle_radius=material.particle_radius,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=material.damping,
    )
    if visual_material is not None:
        spawn_cfg.visual_material = visual_material
    return spawn_cfg
