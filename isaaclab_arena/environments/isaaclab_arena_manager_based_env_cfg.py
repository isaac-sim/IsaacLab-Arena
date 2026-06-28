# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sim import RenderCfg, SimulationCfg

# Import the decorator directly. With the public Isaac Lab wheel,
# `from isaaclab.utils import configclass` can bind the configclass *submodule*
# (not the decorator) once isaaclab.envs.mimic_env_cfg or isaaclab_newton has
# been imported first, making @configclass fail. The direct import is order-safe.
from isaaclab.utils.configclass import configclass

try:
    # Isaac Lab 3.0 (public 3.0.0b2 wheel) houses the MuJoCo-Warp solver cfg here.
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
except ImportError:
    # Older Isaac Lab (the `arena` branch) keeps it in newton_manager_cfg.
    from isaaclab_newton.physics.newton_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class ArenaPhysicsCfg(PresetCfg):
    """Physics backend presets available to all Arena environments.

    ``default`` / ``physx`` use the stock PhysX backend.
    ``newton`` uses MuJoCo-Warp via Newton with solver parameters tuned
    for dexterous manipulation (matches ``KukaAllegroPhysicsCfg.newton``).
    """

    physx = PhysxCfg()
    newton = NewtonCfg(
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
    default = physx


@configclass
class IsaacLabArenaManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-07-29): The following definitions are taken from the base class.
    # scene: InteractiveSceneCfg
    # observations: object
    # actions: object
    # events: object
    # terminations: object
    # recorders: object

    # Kill the unused managers
    commands = None
    rewards = None
    curriculum = None

    metrics: object | None = None

    episode_recorders: object | None = None

    # Task language description
    task_description: str | None = None

    # Override the RTX renderer's built-in scene ambient (carb /rtx/sceneDb/ambientLightIntensity, default 1.0 with
    # color [0.1, 0.1, 0.1]) so that USD light prims fully control scene illumination.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        # Default 2; ILA_RENDER_INTERVAL raises it to render less often (perf lever for the bridge, where
        # GaP only reads cameras at static perceive nodes). Opt-in -- default behaviour is unchanged.
        render_interval=int(os.environ.get("ILA_RENDER_INTERVAL", "2")),
        render=RenderCfg(
            carb_settings={
                "/rtx/sceneDb/ambientLightIntensity": 0.0,
            },
        ),
    )
    decimation: int = 4
    episode_length_s: float = 50.0
    wait_for_textures: bool = False


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []
    pass
