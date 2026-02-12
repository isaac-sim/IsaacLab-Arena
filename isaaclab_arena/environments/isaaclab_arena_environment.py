# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.teleop_device_base import TeleopDeviceBase
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.orchestrator.orchestrator_base import OrchestratorBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase

from isaaclab_arena.reinforment_learning.frameworks import RLFramework


class IsaacLabArenaEnvironment:
    """Describes an environment in IsaacLab Arena."""

    def __init__(
        self,
        name: str,
        scene: Scene,
        embodiment: EmbodimentBase | None = None,
        task: TaskBase | None = None,
        teleop_device: TeleopDeviceBase | None = None,
        orchestrator: OrchestratorBase | None = None,
        env_cfg_callback: Callable[IsaacLabArenaManagerBasedRLEnvCfg] | None = None,
        rl_framework: RLFramework | None = None,
        rl_policy_cfg: str | None = None,
    ):
        """
        Args:
            name: The name of the environment.
            scene: The scene to use in the environment.
            embodiment: The embodiment to use in the environment.
            task: The task to use in the environment.
            teleop_device: The teleop device to use in the environment.
            orchestrator: The orchestrator to use in the environment.
            env_cfg_callback: A callback function that modifies the environment configuration.
        """
        self.name = name
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        self.teleop_device = teleop_device
        self.orchestrator = orchestrator
        self.env_cfg_callback = env_cfg_callback
        self.rl_framework = rl_framework
        self.rl_policy_cfg = rl_policy_cfg
