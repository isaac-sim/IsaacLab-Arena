# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for Arena's physics-backend presets.

Every backend-related decision (which config a preset selects, which simulation backend it runs on,
whether it can simulate soft bodies, and whether it forces ``replicate_physics``) is expressed here
as data in :data:`ARENA_PHYSICS_PRESETS`. Assets and the environment builder consult this registry
by *backend*; neither names a concrete solver variant. Adding a new Newton solver variant is a single
new registry entry -- no asset, builder, or config-class edit.

The preset *name* labels the solver variant; the config *class* describes the backend (mirrors Isaac
Lab's own preset model). Newton variants share a ``newton_`` name prefix, but nothing in Arena keys
off that prefix -- the backend is read from :attr:`PhysicsPreset.backend`.
"""

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
    """The physics backend family a preset runs on."""

    PHYSX = "physx"
    NEWTON = "newton"


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """Newton physics config with global deformable-object model parameters."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


@dataclass(frozen=True)
class PhysicsPreset:
    """Declarative metadata for one physics preset.

    Args:
        name: The preset name (as passed to ``--presets``).
        backend: The simulation backend family this preset runs on.
        cfg: The concrete physics config instance selected by this preset.
        supports_soft_body: Whether Arena supports soft-body (deformable) objects on this preset.
        replicate_physics: ``True`` forces ``scene.replicate_physics`` on, ``None`` leaves the env
            default untouched.
    """

    name: str
    backend: SimulationBackend
    cfg: PhysicsCfg
    supports_soft_body: bool
    replicate_physics: bool | None


# --- Backend config instances (defined once; referenced by the registry and ArenaPhysicsCfg) --------

_PHYSX_CFG = PhysxCfg()

# MuJoCo-Warp via Newton, tuned for dexterous manipulation (matches KukaAllegroPhysicsCfg.newton).
_NEWTON_MJWARP_CFG = NewtonCfg(
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

# Newton rigid (MJWarp) coupled with the VBD soft-body solver -- the only soft-capable preset today.
_NEWTON_MJWARP_VBD_CFG = DeformableNewtonCfg(
    solver_cfg=CoupledMJWarpVBDSolverCfg(
        # Rigid solver settings favor contact-rich manipulation: a high contact budget
        # (njmax/nconmax), elliptic friction cone with large impratio for stable grasps, and many
        # solver/CCD iterations so gripper-object contacts resolve without penetration.
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
        # Self-contact and periodic collision re-detection are off: the objects are convex and
        # small, so the extra cost buys nothing.
        soft_solver_cfg=VBDSolverCfg(
            iterations=10,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",
    ),
    # Contact stiffness (ke), damping (kd), and friction (mu) for soft contacts (deformable) and
    # rigid shape contacts. shape_material_kd damps rigid-body contact: without it, a rigid object
    # resting on the table bounces on its penetration and the high friction turns that bounce into
    # lateral skitter, so it walks off the table; kd=100 lets rigid objects settle and stay put.
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


# --- The registry: one entry per preset name; the single source of truth --------------------------

_PHYSX_PRESET = PhysicsPreset(
    name="physx", backend=SimulationBackend.PHYSX, cfg=_PHYSX_CFG, supports_soft_body=False, replicate_physics=None
)

ARENA_PHYSICS_PRESETS: dict[str, PhysicsPreset] = {
    "physx": _PHYSX_PRESET,
    "newton": PhysicsPreset(
        name="newton",
        backend=SimulationBackend.NEWTON,
        cfg=_NEWTON_MJWARP_CFG,
        supports_soft_body=False,
        replicate_physics=True,
    ),
    "newton_mjwarp_vbd": PhysicsPreset(
        name="newton_mjwarp_vbd",
        backend=SimulationBackend.NEWTON,
        cfg=_NEWTON_MJWARP_VBD_CFG,
        supports_soft_body=True,
        replicate_physics=True,
    ),
    # ``default`` mirrors ``physx`` (the stock backend used when no preset is selected).
    "default": _PHYSX_PRESET,
}

DEFAULT_PRESET = "physx"
"""Preset used for rigid-only scenes when no preset is selected."""

DEFAULT_SOFT_BODY_PRESET = "newton_mjwarp_vbd"
"""Preset auto-selected for soft-body scenes when no preset is selected."""


def preset_backend(name: str) -> SimulationBackend:
    """Return the simulation backend a preset runs on."""
    return ARENA_PHYSICS_PRESETS[name].backend


def soft_body_presets() -> frozenset[str]:
    """Return the set of presets that support soft-body (deformable) objects."""
    return frozenset(name for name, p in ARENA_PHYSICS_PRESETS.items() if p.supports_soft_body)


def is_soft_body_preset(name: str) -> bool:
    """Return whether a preset supports soft-body (deformable) objects."""
    return name in ARENA_PHYSICS_PRESETS and ARENA_PHYSICS_PRESETS[name].supports_soft_body


def _build_arena_physics_cfg_class() -> type[PresetCfg]:
    """Build the ``ArenaPhysicsCfg`` ``PresetCfg`` subclass with one field per registry preset.

    Fields are generated from :data:`ARENA_PHYSICS_PRESETS` (the single source of truth), so adding a
    new preset needs exactly one registry entry and no edit here. Mirrors how ``isaaclab_tasks``'
    ``preset()`` factory builds a ``PresetCfg`` subclass from a dict.
    """
    fields = {name: preset.cfg for name, preset in ARENA_PHYSICS_PRESETS.items()}
    namespace = {
        "__doc__": (
            "Physics backend presets available to all Arena environments (generated from the physics-preset registry)."
        ),
        "__annotations__": {name: type(cfg) for name, cfg in fields.items()},
        **fields,
    }
    return configclass(type("ArenaPhysicsCfg", (PresetCfg,), namespace))


ArenaPhysicsCfg = _build_arena_physics_cfg_class()
"""``PresetCfg`` surface for name-keyed physics-cfg lookup; fields generated from the registry."""
