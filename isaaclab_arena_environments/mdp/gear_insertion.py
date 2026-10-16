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
    """Spawn DROID with convex finger pads and friction over the pad contact patch."""
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

    # Let the finger pads determine compliance when gripping the gear.
    apply_collision_properties(
        f"{prim.GetPath()}/Gripper/.*",
        [MujocoCollisionCfg(priority=2, condim=4, solref=(0.002, 1.0), solimp=(0.95, 0.99, 0.001, 0.5, 2.0))],
    )
    return prim


def gear_insertion_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure Newton contact dynamics and camera recording for gear insertion."""
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_arena.recording.parallel_camera_hdf5_file_handler import ParallelCameraHDF5DatasetFileHandler

    assert isinstance(env_cfg.sim.physics, NewtonCfg), "gear_insertion requires the Newton physics backend"

    # Keep contact integration at 960 Hz without running Python managers at that rate.
    env_cfg.sim.dt = 1.0 / 60.0
    env_cfg.decimation = 2
    env_cfg.sim.render_interval = 2
    env_cfg.sim.physics.num_substeps = 16
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    # The gear and base use convex pieces, including an open bore. MuJoCo-Warp
    # recomputes their contact manifolds at each solver substep.
    env_cfg.sim.physics.solver_cfg.solver = "newton"
    env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    env_cfg.sim.physics.solver_cfg.enable_multiccd = False
    env_cfg.sim.physics.solver_cfg.impratio = 1.0
    env_cfg.sim.physics.solver_cfg.iterations = 50
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 2048
    env_cfg.sim.physics.solver_cfg.nconmax = 512
    env_cfg.sim.physics.collision_cfg = None
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.replicate_physics = True
    env_cfg.scene.robot.spawn.func = _spawn_gear_insertion_droid
    for asset_cfg in (env_cfg.scene.robot, env_cfg.scene.gear_insertion_medium_gear):
        asset_cfg.spawn.physics_material.static_friction = 0.8
        asset_cfg.spawn.physics_material.dynamic_friction = 0.8
    env_cfg.scene.robot.spawn.physics_material.torsional_friction = 0.005
    env_cfg.events.randomize_franka_joint_state = None
    if env_cfg.demo_recorder_config is not None:
        env_cfg.demo_recorder_config.dataset_file_handler_class_type = ParallelCameraHDF5DatasetFileHandler
    return env_cfg
