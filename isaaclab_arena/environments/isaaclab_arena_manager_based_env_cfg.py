# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.utils.configclass import configclass

# Physics presets are defined once in ``physics_presets`` (the single source of truth). Re-exported
# here so existing ``from ...isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg`` importers
# keep working.
from isaaclab_arena.environments.physics_presets import ArenaPhysicsCfg, DeformableNewtonCfg  # noqa: F401


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
        render_interval=2,
        render=RenderCfg(
            carb_settings={
                "/rtx/sceneDb/ambientLightIntensity": 0.0,
                # Workaround for IsaacLab #6424: stop the physx-tensors filter matcher from
                # recursing into leaf collision shapes so a contact filter pointing at a rigid
                # body with multiple collision shapes resolves to a single entry (otherwise the
                # view fails with "expected 1, found N").
                "/physics/tensors/recursiveLeafPatternMatch": False,
            },
        ),
    )
    decimation: int = 4
    wait_for_textures: bool = False


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
