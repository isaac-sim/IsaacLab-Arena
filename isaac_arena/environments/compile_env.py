# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.isaac_arena_manager_based_env import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacArenaManagerBasedRLEnvCfg,
)
from isaac_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaac_arena.utils.configclass import combine_configclass_instances


class ArenaEnvBuilder:
    """Compose Isaac Arena → IsaacLab configs"""

    DEFAULT_SCENE_CFG = InteractiveSceneCfg(num_envs=4096, env_spacing=30.0, replicate_physics=False)

    def __init__(self, arena_env: IsaacArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args

    def orchestrate(self) -> None:
        """Orchestrate the environment member interaction"""
        if self.arena_env.orchestrator is not None:
            self.arena_env.orchestrator.orchestrate(
                self.arena_env.embodiment, self.arena_env.scene, self.arena_env.task
            )

    def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
        """Return base ManagerBased cfg (scene+events+terminations+xr), no registration."""

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        scene_cfg = combine_configclass_instances(
            "SceneCfg",
            self.DEFAULT_SCENE_CFG,
            self.arena_env.scene.get_scene_cfg(),
            self.arena_env.embodiment.get_scene_cfg(),
            self.arena_env.task.get_scene_cfg(),
        )
        observation_cfg = self.arena_env.embodiment.get_observation_cfg()
        events_cfg = combine_configclass_instances(
            "EventsCfg",
            self.arena_env.embodiment.get_events_cfg(),
            self.arena_env.scene.get_events_cfg(),
            self.arena_env.task.get_events_cfg(),
        )
        termination_cfg = combine_configclass_instances(
            "TerminationCfg",
            self.arena_env.task.get_termination_cfg(),
            self.arena_env.scene.get_termination_cfg(),
            self.arena_env.embodiment.get_termination_cfg(),
        )
        actions_cfg = self.arena_env.embodiment.get_action_cfg()
        xr_cfg = self.arena_env.embodiment.get_xr_cfg()
        if self.arena_env.teleop_device is not None:
            teleop_device_cfg = self.arena_env.teleop_device.get_teleop_device_cfg(embodiment=self.arena_env.embodiment)
        else:
            teleop_device_cfg = None
        metrics = self.arena_env.task.get_metrics()
        metrics_recorder_manager_cfg = metrics_to_recorder_manager_cfg(metrics)

        # Base has to be specified explicitly to avoid type errors and not lose inheritance.
        recorder_manager_cfg = combine_configclass_instances(
            "RecorderManagerCfg",
            metrics_recorder_manager_cfg,
            self.arena_env.task.get_recorder_term_cfg(),
            self.arena_env.embodiment.get_recorder_term_cfg(),
            bases=(RecorderManagerBaseCfg,),
        )

        isaac_arena_env = self.arena_env

        # Build the environment configuration
        if not self.args.mimic:
            env_cfg = IsaacArenaManagerBasedRLEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                xr=xr_cfg,
                teleop_devices=teleop_device_cfg,
                recorders=recorder_manager_cfg,
                metrics=metrics,
                isaac_arena_env=isaac_arena_env,
            )
        else:
            task_mimic_env_cfg = self.arena_env.task.get_mimic_env_cfg(embodiment_name=self.arena_env.embodiment.name)
            env_cfg = IsaacArenaManagerBasedMimicEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                xr=xr_cfg,
                teleop_devices=teleop_device_cfg,
                # Mimic stuff
                datagen_config=task_mimic_env_cfg.datagen_config,
                subtask_configs=task_mimic_env_cfg.subtask_configs,
                task_constraint_configs=task_mimic_env_cfg.task_constraint_configs,
                # NOTE(alexmillane, 2025-09-25): Metric + recorders excluded from mimic env,
                # I assume that they're not needed for the mimic env.
                # recorders=recorder_manager_cfg,
                # metrics=metrics,
                isaac_arena_env=isaac_arena_env,
            )
        return env_cfg

    def get_entry_point(self) -> str | type[ManagerBasedRLMimicEnv]:
        """Return the entry point of the environment."""
        if self.args.mimic:
            return self.arena_env.embodiment.get_mimic_env()
        else:
            return "isaaclab.envs:ManagerBasedRLEnv"

    def build_registered(
        self, env_cfg: None | IsaacArenaManagerBasedRLEnvCfg = None
    ) -> tuple[str, IsaacArenaManagerBasedRLEnvCfg]:
        """Register Gym env and parse runtime cfg."""
        name = self.arena_env.name
        # orchestrate the environment member interaction
        self.orchestrate()
        cfg_entry = env_cfg if env_cfg is not None else self.compose_manager_cfg()
        entry_point = self.get_entry_point()
        gym.register(
            id=name,
            entry_point=entry_point,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        cfg = parse_env_cfg(
            name,
            device=self.args.device,
            num_envs=self.args.num_envs,
            use_fabric=not self.args.disable_fabric,
        )
        return name, cfg

    def make_registered(self, env_cfg: None | IsaacArenaManagerBasedRLEnvCfg = None) -> ManagerBasedEnv:
        env, _ = self.make_registered_and_return_cfg(env_cfg)
        return env

    def make_registered_and_return_cfg(
        self, env_cfg: None | IsaacArenaManagerBasedRLEnvCfg = None
    ) -> tuple[ManagerBasedEnv, IsaacArenaManagerBasedRLEnvCfg]:
        name, cfg = self.build_registered(env_cfg)
        return gym.make(name, cfg=cfg).unwrapped, cfg
