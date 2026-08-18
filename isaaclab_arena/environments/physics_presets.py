# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Named physics presets available to Arena environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from isaaclab.physics import PhysicsCfg
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


class SimulationBackend(str, Enum):
    """Physics backend family used by a preset."""

    PHYSX = "physx"
    NEWTON = "newton"


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """Newton physics configuration with global deformable model parameters."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


@dataclass(frozen=True)
class PhysicsPreset:
    """Describe one named physics configuration."""

    physics_cfg: PhysicsCfg
    backend: SimulationBackend
    replicate_physics: bool
    supported_deformable_kinds: frozenset[str]


_PHYSX_CFG = PhysxCfg()

_NEWTON_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=300,
        nconmax=400,
        impratio=10.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=False,
        use_mujoco_contacts=False,
        ccd_iterations=15000,
    ),
    num_substeps=2,
    debug_mode=False,
)

# Proven volume-deformable settings from the Franka soft-lift implementation.
_NEWTON_MJWARP_VBD_CFG = DeformableNewtonCfg(
    solver_cfg=CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=400,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            ccd_iterations=15000,
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=10,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",
    ),
    model_cfg=NewtonModelCfg(
        soft_contact_ke=1.0e4,
        soft_contact_kd=1.0e-5,
        soft_contact_mu=5.0,
        shape_material_ke=4.0e4,
        shape_material_kd=100.0,
        shape_material_mu=5.0,
    ),
    num_substeps=10,
    debug_mode=False,
)


ARENA_PHYSICS_PRESETS: dict[str, PhysicsPreset] = {
    "default": PhysicsPreset(
        physics_cfg=_PHYSX_CFG,
        backend=SimulationBackend.PHYSX,
        replicate_physics=False,
        supported_deformable_kinds=frozenset(),
    ),
    "physx": PhysicsPreset(
        physics_cfg=_PHYSX_CFG,
        backend=SimulationBackend.PHYSX,
        replicate_physics=False,
        supported_deformable_kinds=frozenset(),
    ),
    "newton": PhysicsPreset(
        physics_cfg=_NEWTON_CFG,
        backend=SimulationBackend.NEWTON,
        replicate_physics=True,
        supported_deformable_kinds=frozenset(),
    ),
    "newton_mjwarp_vbd": PhysicsPreset(
        physics_cfg=_NEWTON_MJWARP_VBD_CFG,
        backend=SimulationBackend.NEWTON,
        replicate_physics=True,
        supported_deformable_kinds=frozenset({"volume"}),
    ),
}
"""Registry of supported named physics presets."""


def _build_arena_physics_cfg_class() -> type[PresetCfg]:
    """Build the legacy ``ArenaPhysicsCfg`` API from the registry."""
    fields = {name: preset.physics_cfg for name, preset in ARENA_PHYSICS_PRESETS.items()}
    namespace = {
        "__doc__": "Physics backend presets available to all Arena environments.",
        "__annotations__": {name: type(cfg) for name, cfg in fields.items()},
        **fields,
    }
    return configclass(type("ArenaPhysicsCfg", (PresetCfg,), namespace))


ArenaPhysicsCfg = _build_arena_physics_cfg_class()
"""Legacy name-keyed physics configuration API."""
