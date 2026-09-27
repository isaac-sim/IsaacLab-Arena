# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current CAP cable physics using Arena's native cable import path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_contrib.coupling import NewtonCouplerManager

from .scene import CablePhysics

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

_ACTIVE_PHYSICS: CablePhysics | None = None
_PIN_CABLE_START = True


def _configure_native_cable_builder(builder, world_index: int, *_unused) -> None:
    """Apply CAP's cable friction and fixed-start constraint after native import."""
    physics = _ACTIVE_PHYSICS
    assert physics is not None, "Cable physics must be selected before Newton imports the scene."
    active_world = builder.current_world
    assert active_world in (
        -1,
        world_index,
    ), f"Expected Newton world {world_index}, but the active builder world is {active_world}."

    cable_shapes = []
    for shape_index, (label, shape_world) in enumerate(zip(builder.shape_label, builder.shape_world)):
        if not isinstance(label, str) or int(shape_world) != active_world:
            continue
        cable_path, separator, suffix = label.rpartition("_edge_capsule_")
        if separator and suffix.isdigit() and cable_path.endswith("/Cable/geometry/mesh"):
            cable_shapes.append((int(suffix), shape_index, cable_path))
    cable_shapes.sort()
    assert cable_shapes, f"Unable to find the Arena cable shapes in Newton world {world_index}."
    cable_path = cable_shapes[0][2]
    assert all(
        path == cable_path for _, _, path in cable_shapes
    ), f"Expected one Arena cable in Newton world {world_index}."

    for _, shape_index, _ in cable_shapes:
        builder.shape_material_mu[shape_index] = physics.cable_friction
        builder.shape_material_ke[shape_index] = physics.contact_stiffness
        builder.shape_material_kd[shape_index] = physics.contact_damping
        builder.shape_gap[shape_index] = physics.contact_gap
        builder.shape_margin[shape_index] = 0.0

    # CableMaterialCfg carries elastic moduli but has no damping fields. Restore
    # CAP's direct per-joint rod constants after Arena's native USD import.
    cable_joint_prefix = f"{cable_path}_cable_"
    cable_joint_indices = [
        index
        for index, label in enumerate(builder.joint_label)
        if isinstance(label, str)
        and label.startswith(cable_joint_prefix)
        and int(builder.joint_world[index]) == active_world
    ]
    assert (
        len(cable_joint_indices) == len(cable_shapes) - 1
    ), f"Expected {len(cable_shapes) - 1} Arena cable joints, got {len(cable_joint_indices)}."
    stiffness = (
        physics.stretch_stiffness,
        physics.stretch_stiffness,
        physics.bend_stiffness,
        physics.bend_stiffness,
    )
    damping = (
        physics.stretch_damping,
        physics.stretch_damping,
        physics.bend_damping,
        physics.bend_damping,
    )
    for joint_index in cable_joint_indices:
        assert tuple(builder.joint_dof_dim[joint_index]) == (
            2,
            2,
        ), f"Unexpected Arena cable joint layout: {builder.joint_dof_dim[joint_index]}."
        dof_start = int(builder.joint_qd_start[joint_index])
        builder.joint_target_ke[dof_start : dof_start + 4] = stiffness
        builder.joint_target_kd[dof_start : dof_start + 4] = damping

    if not _PIN_CABLE_START:
        return
    first_shape_index = cable_shapes[0][1]
    first_body_index = int(builder.shape_body[first_shape_index])
    pose = builder.body_q[first_body_index]
    anchor_label = f"{cable_path.rsplit('/geometry/mesh', 1)[0]}/Anchor"
    if anchor_label in builder.joint_label:
        return
    builder.add_joint_fixed(
        parent=-1,
        child=first_body_index,
        parent_xform=wp.transform(
            wp.vec3(*[float(value) for value in pose[:3]]),
            wp.quat(*[float(value) for value in pose[3:7]]),
        ),
        child_xform=wp.transform_identity(),
        label=anchor_label,
    )


class NewtonArenaCableRoutingCouplerManager(NewtonCouplerManager):
    """Install only the task-specific pin/material hook around Arena Cable."""

    @classmethod
    def initialize(cls, sim_context) -> None:
        from isaaclab_newton.physics import NewtonManager

        if _configure_native_cable_builder not in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.append(_configure_native_cable_builder)
        super().initialize(sim_context)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        from isaaclab_newton.physics import NewtonManager

        if hasattr(NewtonManager, "_per_world_builder_hooks"):
            NewtonManager._per_world_builder_hooks = [
                hook for hook in NewtonManager._per_world_builder_hooks if hook is not _configure_native_cable_builder
            ]
        super()._solver_specific_clear()


def configure_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    physics: CablePhysics,
    pin_start: bool,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure the current CAP rigid/VBD solve around an Arena Cable."""
    from isaaclab.utils.configclass import configclass
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
    from isaaclab_newton.physics import (
        MJWarpSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonShapeCfg,
        VBDSolverCfg,
    )
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    global _ACTIVE_PHYSICS, _PIN_CABLE_START
    _ACTIVE_PHYSICS = physics
    _PIN_CABLE_START = pin_start
    assert env_cfg.scene.num_envs == 1, "The current cable-routing compatibility environment supports one env only."

    @configclass
    class CableRoutingShapeCfg(NewtonShapeCfg):
        ke: float = physics.contact_stiffness
        kd: float = physics.contact_damping
        mu: float = physics.fixture_friction

    @configclass
    class CableRoutingVBDSolverCfg(VBDSolverCfg):
        rigid_avbd_beta: float = 1.0e2
        rigid_contact_k_start: float = 1.0e3
        # CAP's manual manager allocates this state before CUDA graph capture.
        # Arena Cable uses the standard manager lifecycle, so keep history off.
        rigid_contact_history: bool = False
        rigid_body_contact_buffer_size: int = 256

    collision_cfg = NewtonCollisionPipelineCfg()
    env_cfg.sim.dt = 1.0 / 60.0
    env_cfg.sim.render_interval = 4
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=physics.fixture_friction,
        dynamic_friction=physics.fixture_friction,
        restitution=0.0,
        contact_stiffness=physics.contact_stiffness,
        contact_damping=physics.contact_damping,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            class_type=NewtonArenaCableRoutingCouplerManager,
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        njmax=1024,
                        nconmax=512,
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                        impratio=10.0,
                        tolerance=1.0e-8,
                        use_mujoco_contacts=False,
                        disable_sensors=True,
                    ),
                    bodies=[
                        r"/World/envs/env_.*/Yam",
                        r"/World/envs/env_.*/(Board|Peg[0-3]|Anchor|Port)",
                    ],
                    include_body_shapes=False,
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=CableRoutingVBDSolverCfg(iterations=40),
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[
                        r"/World/envs/env_.*/Yam/Geometry/left_base/left_link1/left_link2/left_link3/left_link4/left_link5/left_link6",
                        r"/World/envs/env_.*/Yam/Geometry/right_base/right_link1/right_link2/right_link3/right_link4/right_link5/right_link6",
                        r"/World/envs/env_.*/(Board|Peg[0-3]|Anchor|Port)",
                    ],
                    mode="staggered",
                    mass_scale=20.0,
                    collide_interval=1,
                    collision_pipeline=collision_cfg,
                )
            ],
            iterations=2,
        ),
        default_shape_cfg=CableRoutingShapeCfg(margin=0.0, gap=0.002),
        collision_cfg=collision_cfg,
        num_substeps=16,
        use_cuda_graph=True,
        debug_mode=False,
        deterministic_mode="not_guaranteed",
    )
    env_cfg.decimation = 1
    env_cfg.scene.replicate_physics = True
    return env_cfg


__all__ = ["configure_cable_routing_physics"]
