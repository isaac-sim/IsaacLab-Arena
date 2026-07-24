# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry + backend-neutral deformable-spawn adapter tests (no SimulationApp required).

These lock in that the physics-preset registry is the single source of truth (drift guard + metadata
matrix) and that the backend-neutral spawn adapter reproduces the previous hand-tuned per-object
spawn constants exactly.
"""


def test_registry_matches_arena_physics_cfg_fields() -> None:
    """ArenaPhysicsCfg fields and the registry keys are the same set (drift guard)."""
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg
    from isaaclab_arena.environments.physics_presets import ARENA_PHYSICS_PRESETS

    assert set(ArenaPhysicsCfg.__dataclass_fields__) == set(ARENA_PHYSICS_PRESETS)
    # Each field exposes the registry's cfg by value (configclass deep-copies defaults per instance,
    # which preserves per-build isolation -- so compare by equality, not identity).
    cfg = ArenaPhysicsCfg()
    for name, preset in ARENA_PHYSICS_PRESETS.items():
        assert getattr(cfg, name) == preset.cfg


def test_registry_metadata_matrix() -> None:
    """Each preset's backend / soft-capability / replicate metadata matches the intended behavior."""
    from isaaclab_arena.environments.physics_presets import (
        ARENA_PHYSICS_PRESETS,
        DEFAULT_PRESET,
        DEFAULT_SOFT_BODY_PRESET,
        SimulationBackend,
        is_soft_body_preset,
        soft_body_presets,
    )

    expected = {
        # name: (backend, supports_soft_body, replicate_physics)
        "physx": (SimulationBackend.PHYSX, False, None),
        "newton": (SimulationBackend.NEWTON, False, True),
        "newton_mjwarp_vbd": (SimulationBackend.NEWTON, True, True),
        "default": (SimulationBackend.PHYSX, False, None),
    }
    assert set(ARENA_PHYSICS_PRESETS) == set(expected)
    for name, (backend, soft, replicate) in expected.items():
        preset = ARENA_PHYSICS_PRESETS[name]
        assert preset.backend is backend
        assert preset.supports_soft_body is soft
        assert preset.replicate_physics is replicate

    assert soft_body_presets() == frozenset({"newton_mjwarp_vbd"})
    assert is_soft_body_preset("newton_mjwarp_vbd") and not is_soft_body_preset("physx")
    assert DEFAULT_PRESET == "physx" and DEFAULT_SOFT_BODY_PRESET == "newton_mjwarp_vbd"
    # default mirrors physx (same cfg instance)
    assert ARENA_PHYSICS_PRESETS["default"].cfg is ARENA_PHYSICS_PRESETS["physx"].cfg


def test_backend_object_preset_soft_only_fields() -> None:
    """A soft-body object preset carries exactly the soft presets plus a soft ``default``."""
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.deformable_spawn import backend_object_preset
    from isaaclab_arena.environments.physics_presets import SimulationBackend, soft_body_presets

    marker = {SimulationBackend.PHYSX: "physx-cfg", SimulationBackend.NEWTON: "newton-cfg"}
    preset_cfg = backend_object_preset(lambda backend: marker[backend], soft_body_only=True)

    fields = set(preset_cfg.__dataclass_fields__)
    assert fields == soft_body_presets() | {"default"}
    # Every soft preset (and default) maps to the Newton build; no PhysX field exists.
    for name in fields:
        assert resolve_presets(preset_cfg, selected=(name,)) == "newton-cfg"


def _reference_spawns(usd_path, youngs, poissons, density, physx_tuning, newton_particle_radius, color):
    """Rebuild the pre-refactor per-object spawn constants for equivalence checking."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    from isaaclab_arena.assets.deformable_spawn import lame_parameters

    k_mu, k_lambda = lame_parameters(youngs, poissons)
    physx = UsdFileCfg(
        usd_path=usd_path,
        deformable_props=PhysxDeformableBodyPropertiesCfg(**physx_tuning),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=PhysxDeformableBodyMaterialCfg(poissons_ratio=poissons, youngs_modulus=youngs),
    )
    newton = UsdFileCfg(
        usd_path=usd_path,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=NewtonDeformableBodyMaterialCfg(
            density=density, particle_radius=newton_particle_radius, k_mu=k_mu, k_lambda=k_lambda
        ),
    )
    return physx, newton


def test_build_deformable_spawn_matches_reference_constants() -> None:
    """The adapter reproduces the previous hand-tuned spawn for BOTH library objects, field-by-field."""
    from isaaclab_arena.assets.deformable_spawn import build_deformable_spawn
    from isaaclab_arena.assets.object_library import (
        _DEFORMABLE_CUBE_TET_USD,
        _DEFORMABLE_SPHERE_TET_USD,
        _PROCEDURAL_DEFORMABLE_CUBE_MATERIAL,
        _PROCEDURAL_DEFORMABLE_SPHERE_MATERIAL,
    )
    from isaaclab_arena.environments.physics_presets import SimulationBackend

    cases = (
        (
            _DEFORMABLE_SPHERE_TET_USD,
            _PROCEDURAL_DEFORMABLE_SPHERE_MATERIAL,
            _reference_spawns(
                _DEFORMABLE_SPHERE_TET_USD,
                1.0e5,
                0.4,
                300.0,
                dict(rest_offset=0.0, contact_offset=0.002, solver_position_iteration_count=16, linear_damping=0.01),
                0.008,
                (0.9, 0.25, 0.2),
            ),
        ),
        (
            _DEFORMABLE_CUBE_TET_USD,
            _PROCEDURAL_DEFORMABLE_CUBE_MATERIAL,
            _reference_spawns(
                _DEFORMABLE_CUBE_TET_USD,
                2.0e5,
                0.4,
                300.0,
                dict(rest_offset=0.0, contact_offset=0.001, solver_position_iteration_count=24, linear_damping=0.02),
                0.006,
                (0.12, 0.28, 0.85),
            ),
        ),
    )

    for usd_path, material, (ref_physx, ref_newton) in cases:
        visual = ref_physx.visual_material
        built_physx = build_deformable_spawn(usd_path, material, SimulationBackend.PHYSX, visual_material=visual)
        built_newton = build_deformable_spawn(usd_path, material, SimulationBackend.NEWTON, visual_material=visual)
        for built, ref in ((built_physx, ref_physx), (built_newton, ref_newton)):
            assert built.usd_path == ref.usd_path
            assert built.deformable_props.to_dict() == ref.deformable_props.to_dict()
            assert built.physics_material.to_dict() == ref.physics_material.to_dict()
