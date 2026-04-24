# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.metrics.metric_base import MetricBase


# Default physics material for rigid bodies that don't have one explicitly
# bound. Applies to the table (scene USD materials are visual-only) and the
# ground plane. Defaults are 0.5/0.5 with "average" combine, which let objects
# slide on the table after the small t=0 drop. Bumping to 0.8/0.6 with "max"
# means an object with its own explicit friction keeps its value (max wins),
# while material-less surfaces get decent grip.
_DEFAULT_PHYSICS_MATERIAL = RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)


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

    # Metrics
    metrics: list[MetricBase] | None = None

    # Isaaclab Arena Env. Held as a member to allow use of internal functions
    isaaclab_arena_env: IsaacLabArenaEnvironment | None = None

    # Overriding defaults from base class
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=2,
        physics_material=_DEFAULT_PHYSICS_MATERIAL,
    )
    decimation: int = 4
    episode_length_s: float = 50.0
    wait_for_textures: bool = False
    num_renders_on_reset: int = 5


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an IsaacLab Arena environment."""

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []
    pass
