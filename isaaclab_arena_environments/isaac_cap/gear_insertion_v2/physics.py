# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared physics defaults for industrial benchmark environments."""

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg


def disable_mjwarp_sensors(physics: NewtonCfg) -> NewtonCfg:
    """Turn off MJWarp's internal sensors, including in coupled rigid solvers.

    Arena reads Newton state, so MJWarp's own sensors are unsupported and have
    to stay off. Walks nested solver entries so a coupled configuration is
    covered as well as a plain one, and leaves every other solver field alone.
    """
    solvers = [physics.solver_cfg]
    while solvers:
        solver = solvers.pop()
        if isinstance(solver, MJWarpSolverCfg):
            solver.disable_sensors = True
        solvers.extend(entry.solver_cfg for entry in getattr(solver, "entries", ()))
    return physics
