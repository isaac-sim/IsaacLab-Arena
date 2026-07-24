# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral deformable spawn construction.

This module is the *only* place in Arena that names the concrete PhysX / Newton deformable config
classes. Deformable assets declare a backend-neutral :class:`DeformableMaterial` (shared physical
properties plus small per-backend solver-tuning structs) and let :func:`build_deformable_spawn`
translate it into the spawn config for a given :class:`~isaaclab_arena.environments.physics_presets.SimulationBackend`.

:func:`backend_object_preset` then fans a single per-backend spawn builder out to every physics
preset in the registry, so adding a new Newton solver variant needs no change here or in any asset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_arena.environments.physics_presets import (
    ARENA_PHYSICS_PRESETS,
    DEFAULT_PRESET,
    DEFAULT_SOFT_BODY_PRESET,
    SimulationBackend,
    preset_backend,
    soft_body_presets,
)


def lame_parameters(youngs_modulus: float, poissons_ratio: float) -> tuple[float, float]:
    """Convert (Young's modulus, Poisson's ratio) to Lamé (mu, lambda).

    Newton's material takes Lamé parameters directly, whereas PhysX takes Young's/Poisson's; deriving
    both from the same pair keeps the two backends materially equivalent.

    Args:
        youngs_modulus: Young's modulus [Pa].
        poissons_ratio: Poisson's ratio.

    Returns:
        The Lamé parameters ``(k_mu, k_lambda)`` [Pa].
    """
    k_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    k_lambda = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    return k_mu, k_lambda


@dataclass
class PhysxDeformableTuning:
    """PhysX-specific FEM solver knobs (no backend-neutral equivalent)."""

    rest_offset: float = 0.0
    contact_offset: float = 0.002
    solver_position_iteration_count: int = 16
    linear_damping: float = 0.01


@dataclass
class NewtonDeformableTuning:
    """Newton-specific VBD solver knobs (no backend-neutral equivalent).

    ``particle_radius`` is the VBD collision radius [m]; keep it below the tet edge length so
    neighboring particles collide against the gripper rather than tunnel through.
    """

    particle_radius: float = 0.008


@dataclass
class DeformableMaterial:
    """Backend-neutral deformable material.

    The shared physical properties (Young's/Poisson/density) are declared once and converted per
    backend; the ``physx`` / ``newton`` sub-structs carry only the genuinely backend-specific solver
    knobs. Set the sub-structs per object -- their defaults are not tuned for any particular asset.

    Args:
        youngs_modulus: Young's modulus [Pa] (stiffness).
        poissons_ratio: Poisson's ratio.
        density: Material density [kg/m^3] (consumed by the Newton backend; PhysX uses its default).
        physx: PhysX-specific solver tuning.
        newton: Newton-specific solver tuning.
    """

    youngs_modulus: float
    poissons_ratio: float
    density: float
    physx: PhysxDeformableTuning = field(default_factory=PhysxDeformableTuning)
    newton: NewtonDeformableTuning = field(default_factory=NewtonDeformableTuning)


def build_deformable_spawn(
    usd_path: str,
    material: DeformableMaterial,
    backend: SimulationBackend,
    *,
    visual_material: VisualMaterialCfg,
) -> UsdFileCfg:
    """Build the deformable spawn config for a backend from a backend-neutral material.

    Args:
        usd_path: Path to the pre-tetrahedralized deformable USD asset.
        material: The backend-neutral material.
        backend: The simulation backend to build the spawn for.
        visual_material: The visual (render) material for the object.
    """
    if backend is SimulationBackend.PHYSX:
        return UsdFileCfg(
            usd_path=usd_path,
            deformable_props=PhysxDeformableBodyPropertiesCfg(
                rest_offset=material.physx.rest_offset,
                contact_offset=material.physx.contact_offset,
                solver_position_iteration_count=material.physx.solver_position_iteration_count,
                linear_damping=material.physx.linear_damping,
            ),
            visual_material=visual_material,
            physics_material=PhysxDeformableBodyMaterialCfg(
                poissons_ratio=material.poissons_ratio,
                youngs_modulus=material.youngs_modulus,
            ),
        )
    if backend is SimulationBackend.NEWTON:
        k_mu, k_lambda = lame_parameters(material.youngs_modulus, material.poissons_ratio)
        return UsdFileCfg(
            usd_path=usd_path,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=visual_material,
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=material.density,
                particle_radius=material.newton.particle_radius,
                k_mu=k_mu,
                k_lambda=k_lambda,
            ),
        )
    raise ValueError(f"Unsupported simulation backend for deformable spawn: {backend}")


def backend_object_preset(
    make_object_cfg: Callable[[SimulationBackend], AssetBaseCfg],
    *,
    soft_body_only: bool = False,
) -> PresetCfg:
    """Fan a per-backend object-cfg builder out to every registry preset of that backend.

    The caller supplies a builder keyed by *backend* (2 values); this covers all present and future
    solver *variants* of each backend by reading the physics-preset registry, so no asset ever names
    a variant. The resulting ``PresetCfg`` is resolved by Isaac Lab's ``resolve_presets`` like any
    other preset node.

    Args:
        make_object_cfg: Builds the object's scene config for a given backend.
        soft_body_only: When True, restrict the preset fields to soft-body-capable presets and use
            the soft-body default; otherwise cover all presets and use the stock default.
    """
    names = soft_body_presets() if soft_body_only else set(ARENA_PHYSICS_PRESETS)
    fields: dict[str, AssetBaseCfg] = {name: make_object_cfg(preset_backend(name)) for name in names}
    default_name = DEFAULT_SOFT_BODY_PRESET if soft_body_only else DEFAULT_PRESET
    fields["default"] = make_object_cfg(preset_backend(default_name))
    return preset(**fields)
