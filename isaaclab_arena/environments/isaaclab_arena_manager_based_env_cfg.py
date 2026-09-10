# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

# Import from the package root so this resolves whether MJWarpSolverCfg lives in
# newton_manager_cfg (older isaaclab_newton) or mjwarp_manager_cfg (Isaac Lab Beta 2).
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings
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

    demo_recorder_config: RecorderManagerBaseCfg | None = None
    """Recorder configuration used by demonstration collection scripts."""

    # Task language description
    task_description: str | None = None

    # Control rate: sim.dt (1/120 s) x decimation (8) = 15 Hz
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,
    )
    decimation: int = 8
    wait_for_textures: bool = False


def apply_arena_global_settings() -> None:
    """Apply Arena's process-global RTX and physics settings before environment construction."""
    apply_isaac_rtx_global_settings(
        IsaacRtxRendererGlobalSettingsCfg(
            # Disable RTX's built-in scene ambient so USD light prims fully control illumination.
            ambient_light_intensity=0.0,
            carb_settings={
                # Work around IsaacLab #6424: do not recurse into leaf collision shapes when
                # resolving a contact filter that targets a multi-shape rigid body.
                "/physics/tensors/recursiveLeafPatternMatch": False,
            },
        )
    )


def set_control_rate_50hz(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Set 50 Hz control (sim dt 1/200, decimation 4), Arena's pre-15 Hz default rate.

    Args:
        env_cfg: The environment configuration to modify in place.

    Returns:
        The same configuration, so this can be used directly as an ``env_cfg_callback``.
    """
    env_cfg.sim.dt = 1 / 200
    env_cfg.decimation = 4
    return env_cfg


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []

    # Data generation keeps the longer historical default so demos are not truncated; the task's
    # (shorter) episode length is only applied to non-mimic RL/eval envs by the env builder.
    episode_length_s: float = 50.0
