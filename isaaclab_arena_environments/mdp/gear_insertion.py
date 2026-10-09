# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton physics configuration for the gear insertion environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.utils import clone

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


@clone
def _spawn_gear_insertion_droid(prim_path, spawner_cfg, translation=None, orientation=None, **kwargs):
    """Spawn DROID with smooth fingertip colliders and compliant gripping contacts."""
    from isaaclab.sim.schemas import apply_collision_properties
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.embodiments.droid.droid import spawn_newton_droid

    prim = spawn_newton_droid.__wrapped__(
        prim_path, spawner_cfg, translation=translation, orientation=orientation, **kwargs
    )
    # Use one convex hull per gripper pad for better contact with gear teeth.
    pad_count = 0
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh) and "fingertipsstep" in child.GetName():
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set("convexHull")
            pad_count += 1
    assert pad_count == 2, f"Expected two DROID fingertip meshes, found {pad_count}"

    # Set gripper contact priority to 2 supersede other asset contacts.
    apply_collision_properties(
        f"{prim.GetPath()}/Gripper/.*",
        [MujocoCollisionCfg(priority=2, solref=(0.01, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0))],
    )
    return prim


def gear_insertion_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCfg, NewtonCollisionPipelineCfg

    assert isinstance(env_cfg.sim.physics, NewtonCfg), "gear_insertion requires the Newton physics backend"

    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 8
    env_cfg.sim.render_interval = 8
    env_cfg.sim.physics.num_substeps = 4
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 4096
    env_cfg.sim.physics.solver_cfg.nconmax = 4096
    env_cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(reduce_contacts=True, normal_matching=True)
    )
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.robot.spawn.func = _spawn_gear_insertion_droid
    env_cfg.events.randomize_franka_joint_state = None
    return env_cfg
