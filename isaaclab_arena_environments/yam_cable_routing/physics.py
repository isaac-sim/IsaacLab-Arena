# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coupled Newton physics configuration for YAM cable routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.assets.object_library import YAM_CABLE_ROUTING_CONTACT_DAMPING as CONTACT_DAMPING
from isaaclab_arena.assets.object_library import YAM_CABLE_ROUTING_CONTACT_GAP as CONTACT_GAP
from isaaclab_arena.assets.object_library import YAM_CABLE_ROUTING_CONTACT_STIFFNESS as CONTACT_STIFFNESS
from isaaclab_arena.assets.object_library import make_yam_cable_routing_fixture_material as make_fixture_material

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

CABLE_CONTACT_FRICTION = 0.1
COLLISION_SUBSTEP_INTERVAL = 2


def configure_yam_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure coupled MJWarp rigid-body and VBD cable physics."""
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
                        r"/World/envs/env_.*/YamLeft",
                        r"/World/envs/env_.*/YamRight",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg0",
                        r"/World/envs/env_.*/Peg1",
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
                        r"/World/envs/env_.*/Yam(Left|Right)/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg(0|1)",
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
