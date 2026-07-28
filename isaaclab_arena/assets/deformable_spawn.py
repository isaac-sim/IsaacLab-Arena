# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral deformable spawn construction.

This module is the only place in Arena that names concrete PhysX / Newton deformable config
classes. Arena assets declare a deformable source (USD or mesh spawner) plus a material kind
(volume, surface, or cable-as-surface), and this module translates that declaration into the
backend-specific Isaac Lab spawn config.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg, PhysxSurfaceDeformableBodyMaterialCfg
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_arena.environments.physics_presets import (
    ARENA_PHYSICS_PRESETS,
    DEFAULT_PRESET,
    DEFAULT_SOFT_BODY_PRESET,
    SimulationBackend,
    preset_backend,
    soft_body_presets,
)


class DeformableKind(str, Enum):
    """Topological kind of deformable simulated by Isaac Lab."""

    VOLUME = "volume"
    SURFACE = "surface"
    CABLE = "cable"


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
    """PhysX-specific deformable body and material knobs."""

    rest_offset: float = 0.0
    contact_offset: float = 0.002
    solver_position_iteration_count: int = 16
    linear_damping: float = 0.01
    static_friction: float = 0.25
    dynamic_friction: float = 0.25
    elasticity_damping: float = 0.005


@dataclass
class NewtonDeformableTuning:
    """Newton-specific volume-deformable material knobs.

    ``particle_radius`` is the VBD collision radius [m]; keep it below the tet edge length so
    neighboring particles collide against the gripper rather than tunnel through.
    """

    particle_radius: float = 0.008
    k_damp: float = 0.0


@dataclass
class PhysxSurfaceDeformableTuning:
    """PhysX-specific surface-deformable material and body knobs."""

    rest_offset: float = 0.0
    contact_offset: float = 0.002
    solver_position_iteration_count: int = 16
    linear_damping: float = 0.01
    density: float = 1000.0
    static_friction: float = 0.25
    dynamic_friction: float = 0.25
    youngs_modulus: float = 1.0e6
    poissons_ratio: float = 0.45
    elasticity_damping: float = 0.005
    surface_thickness: float = 0.01
    surface_stretch_stiffness: float = 0.0
    surface_shear_stiffness: float = 0.0
    surface_bend_stiffness: float = 0.0
    bend_damping: float = 0.0


@dataclass
class NewtonSurfaceDeformableTuning:
    """Newton-specific surface-deformable material knobs."""

    density: float = 50.0
    particle_radius: float = 0.005
    tri_ke: float = 5.0e2
    tri_ka: float = 5.0e2
    tri_kd: float = 1.0e-3
    edge_ke: float = 2.0
    edge_kd: float = 1.0e-3


@dataclass
class VolumeDeformableMaterial:
    """Backend-neutral volume deformable material.

    The shared physical properties (Young's/Poisson/density) are declared once and converted per
    backend; the ``physx`` / ``newton`` sub-structs carry only the genuinely backend-specific solver
    knobs. Set the sub-structs per object -- their defaults are not tuned for any particular asset.

    Args:
        youngs_modulus: Young's modulus [Pa] (stiffness).
        poissons_ratio: Poisson's ratio.
        density: Material density [kg/m^3].
        physx: PhysX-specific solver tuning.
        newton: Newton-specific solver tuning.
    """

    youngs_modulus: float
    poissons_ratio: float
    density: float
    kind: DeformableKind = DeformableKind.VOLUME
    physx: PhysxDeformableTuning = field(default_factory=PhysxDeformableTuning)
    newton: NewtonDeformableTuning = field(default_factory=NewtonDeformableTuning)

    def __post_init__(self) -> None:
        self.kind = DeformableKind(self.kind)
        assert self.kind is DeformableKind.VOLUME, f"VolumeDeformableMaterial kind must be volume, got {self.kind!r}."


@dataclass
class SurfaceDeformableMaterial:
    """Backend-neutral surface deformable material.

    Surface materials are intentionally separate from volume materials because Isaac Lab exposes
    different physical controls for cloth-like deformables in Newton and PhysX.
    """

    kind: DeformableKind = DeformableKind.SURFACE
    physx: PhysxSurfaceDeformableTuning = field(default_factory=PhysxSurfaceDeformableTuning)
    newton: NewtonSurfaceDeformableTuning = field(default_factory=NewtonSurfaceDeformableTuning)

    def __post_init__(self) -> None:
        self.kind = DeformableKind(self.kind)
        assert self.kind in (
            DeformableKind.SURFACE,
            DeformableKind.CABLE,
        ), f"SurfaceDeformableMaterial kind must be surface or cable, got {self.kind!r}."


@dataclass
class CableDeformableMaterial(SurfaceDeformableMaterial):
    """Cable-like deformable represented as a narrow Newton/PhysX surface mesh strip."""

    kind: DeformableKind = DeformableKind.CABLE


DeformableMaterial: TypeAlias = VolumeDeformableMaterial
DeformableSource: TypeAlias = str | DeformableObjectSpawnerCfg


def _build_deformable_props(
    material: VolumeDeformableMaterial | SurfaceDeformableMaterial,
    backend: SimulationBackend,
) -> PhysxDeformableBodyPropertiesCfg | NewtonDeformableBodyPropertiesCfg:
    if backend is SimulationBackend.PHYSX:
        tuning = material.physx
        return PhysxDeformableBodyPropertiesCfg(
            rest_offset=tuning.rest_offset,
            contact_offset=tuning.contact_offset,
            solver_position_iteration_count=tuning.solver_position_iteration_count,
            linear_damping=tuning.linear_damping,
        )
    if backend is SimulationBackend.NEWTON:
        return NewtonDeformableBodyPropertiesCfg()
    raise ValueError(f"Unsupported simulation backend for deformable props: {backend}")


def _build_volume_material(
    material: VolumeDeformableMaterial,
    backend: SimulationBackend,
) -> PhysxDeformableBodyMaterialCfg | NewtonDeformableBodyMaterialCfg:
    if backend is SimulationBackend.PHYSX:
        return PhysxDeformableBodyMaterialCfg(
            density=material.density,
            poissons_ratio=material.poissons_ratio,
            youngs_modulus=material.youngs_modulus,
            static_friction=material.physx.static_friction,
            dynamic_friction=material.physx.dynamic_friction,
            elasticity_damping=material.physx.elasticity_damping,
        )
    if backend is SimulationBackend.NEWTON:
        k_mu, k_lambda = lame_parameters(material.youngs_modulus, material.poissons_ratio)
        return NewtonDeformableBodyMaterialCfg(
            density=material.density,
            particle_radius=material.newton.particle_radius,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=material.newton.k_damp,
        )
    raise ValueError(f"Unsupported simulation backend for volume material: {backend}")


def _build_surface_material(
    material: SurfaceDeformableMaterial,
    backend: SimulationBackend,
) -> PhysxSurfaceDeformableBodyMaterialCfg | NewtonSurfaceDeformableBodyMaterialCfg:
    if backend is SimulationBackend.PHYSX:
        tuning = material.physx
        return PhysxSurfaceDeformableBodyMaterialCfg(
            density=tuning.density,
            static_friction=tuning.static_friction,
            dynamic_friction=tuning.dynamic_friction,
            youngs_modulus=tuning.youngs_modulus,
            poissons_ratio=tuning.poissons_ratio,
            elasticity_damping=tuning.elasticity_damping,
            surface_thickness=tuning.surface_thickness,
            surface_stretch_stiffness=tuning.surface_stretch_stiffness,
            surface_shear_stiffness=tuning.surface_shear_stiffness,
            surface_bend_stiffness=tuning.surface_bend_stiffness,
            bend_damping=tuning.bend_damping,
        )
    if backend is SimulationBackend.NEWTON:
        tuning = material.newton
        return NewtonSurfaceDeformableBodyMaterialCfg(
            density=tuning.density,
            particle_radius=tuning.particle_radius,
            tri_ke=tuning.tri_ke,
            tri_ka=tuning.tri_ka,
            tri_kd=tuning.tri_kd,
            edge_ke=tuning.edge_ke,
            edge_kd=tuning.edge_kd,
        )
    raise ValueError(f"Unsupported simulation backend for surface material: {backend}")


def _build_physics_material(
    material: VolumeDeformableMaterial | SurfaceDeformableMaterial,
    backend: SimulationBackend,
) -> (
    PhysxDeformableBodyMaterialCfg
    | NewtonDeformableBodyMaterialCfg
    | PhysxSurfaceDeformableBodyMaterialCfg
    | NewtonSurfaceDeformableBodyMaterialCfg
):
    if material.kind is DeformableKind.VOLUME:
        assert isinstance(material, VolumeDeformableMaterial)
        return _build_volume_material(material, backend)
    assert isinstance(material, SurfaceDeformableMaterial)
    return _build_surface_material(material, backend)


def build_deformable_spawn(
    source: DeformableSource,
    material: VolumeDeformableMaterial | SurfaceDeformableMaterial,
    backend: SimulationBackend,
    *,
    visual_material: VisualMaterialCfg,
    scale: tuple[float, float, float] | None = None,
) -> DeformableObjectSpawnerCfg:
    """Build the deformable spawn config for a backend from a backend-neutral material.

    Args:
        source: Path to a deformable USD asset or an Isaac Lab mesh spawner config.
        material: The backend-neutral material.
        backend: The simulation backend to build the spawn for.
        visual_material: The visual (render) material for the object.
        scale: Optional USD scale. Ignored for mesh spawner sources.
    """
    deformable_props = _build_deformable_props(material, backend)
    physics_material = _build_physics_material(material, backend)

    if isinstance(source, str):
        return UsdFileCfg(
            usd_path=source,
            scale=scale,
            deformable_props=deformable_props,
            visual_material=visual_material,
            physics_material=physics_material,
        )

    spawn_cfg = copy.deepcopy(source)
    spawn_cfg.deformable_props = deformable_props
    spawn_cfg.visual_material = visual_material
    spawn_cfg.physics_material = physics_material
    return spawn_cfg


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
