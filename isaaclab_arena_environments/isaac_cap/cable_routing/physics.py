# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Newton coupling configuration for Isaac Cap cable routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

CONTACT_STIFFNESS = 4.0e4
CONTACT_DAMPING = 1.0e-5
CONTACT_GAP = 0.001
CABLE_CONTACT_FRICTION = 0.1
FIXTURE_CONTACT_FRICTION = 0.5
COLLISION_SUBSTEP_INTERVAL = 2


def make_fixture_material():
    """Create the Newton material shared by cable-routing fixtures."""
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    return NewtonMaterialPropertiesCfg(
        static_friction=FIXTURE_CONTACT_FRICTION,
        dynamic_friction=FIXTURE_CONTACT_FRICTION,
        restitution=0.0,
        contact_stiffness=CONTACT_STIFFNESS,
        contact_damping=CONTACT_DAMPING,
    )


def configure_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure public MJWarp-to-VBD proxy coupling for the task."""
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg

    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.sim.render_interval = 4
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics_material = make_fixture_material()
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        njmax=300,
                        nconmax=200,
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                        ccd_iterations=100,
                    ),
                    bodies=[
                        r"/World/envs/env_.*/LeftRobot",
                        r"/World/envs/env_.*/RightRobot",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg(0|1|2)",
                    ],
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[
                        r"/World/envs/env_.*/(LeftRobot|RightRobot)/Geometry/arm/"
                        r"link_1/link_2/link_3/link_4/link_5/link_6",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg(0|1|2)",
                    ],
                    mode="lagged",
                    mass_scale=1.0,
                    collide_interval=COLLISION_SUBSTEP_INTERVAL,
                )
            ],
            iterations=1,
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=CONTACT_STIFFNESS,
            kd=CONTACT_DAMPING,
            mu=CABLE_CONTACT_FRICTION,
            margin=0.0,
            gap=CONTACT_GAP,
        ),
        num_substeps=10,
        use_cuda_graph=True,
        debug_mode=False,
    )
    env_cfg.decimation = 4
    env_cfg.scene.replicate_physics = True
    return env_cfg
