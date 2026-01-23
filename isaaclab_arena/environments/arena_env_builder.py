# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_arena.assets.asset_registry import DeviceRegistry
from isaaclab_arena.assets.object import Object
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.utils.configclass import combine_configclass_instances


class ArenaEnvBuilder:
    """Compose IsaacLab Arena → IsaacLab configs"""

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args
        self.interactive_scene_cfg = InteractiveSceneCfg(
            num_envs=args.num_envs, env_spacing=args.env_spacing, replicate_physics=False
        )

    def orchestrate(self) -> None:
        """Orchestrate the environment member interaction"""
        if self.arena_env.orchestrator is not None:
            self.arena_env.orchestrator.orchestrate(
                self.arena_env.embodiment, self.arena_env.scene, self.arena_env.task
            )

    def _get_objects_with_relations(self) -> list[Object]:
        """Get all objects from the scene that have relations.

        Returns:
            List of Object instances that have at least one relation.
        """
        objects_with_relations: list[Object] = []
        for asset in self.arena_env.scene.assets.values():
            # TODO(cvolk): Support ObjectSets.
            if not isinstance(asset, Object):
                continue
            if asset.get_relations():
                objects_with_relations.append(asset)
        return objects_with_relations

    def _solve_relations(self) -> None:
        """Solve spatial relations for objects in the scene.

        This method:
        1. Collects all objects from the scene that have relations
        2. Finds the anchor object (marked with IsAnchor)
        3. Runs the ObjectPlacer to solve spatial constraints
        4. Applies solved positions to objects
        """
        # All objects with relations are subjects of the relation solving.
        objects_with_relations = self._get_objects_with_relations()

        if not objects_with_relations:
            print("No objects with relations found in scene. Skipping relation solving.")
            return

        # Run the ObjectPlacer
        placer = ObjectPlacer()
        result = placer.place(objects=objects_with_relations)

        if result.success:
            print(f"Relation solving succeeded after {result.attempts} attempt(s)")
        else:
            print(f"Relation solving not completed after {result.attempts} attempt(s)")

    # This method gives the arena environment a chance to modify the environment configuration.
    # This is a workaround to allow user to gradually move to the new configuration system.
    # THE ORDER MATTERS HERE.
    # THIS WILL BE REMOVED IN THE FUTURE.
    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Modify the environment configuration."""
        env_cfg = self.arena_env.task.modify_env_cfg(env_cfg)
        env_cfg = self.arena_env.embodiment.modify_env_cfg(env_cfg)
        env_cfg = self.arena_env.scene.modify_env_cfg(env_cfg)
        return env_cfg

    def compose_manager_cfg(self) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Return base ManagerBased cfg (scene+events+terminations+xr), no registration."""

        # Solve relations before building scene config so positions are captured correctly.
        if self.args.solve_relations:
            self._solve_relations()

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        scene_cfg = combine_configclass_instances(
            "SceneCfg",
            self.interactive_scene_cfg,
            self.arena_env.scene.get_scene_cfg(),
            self.arena_env.embodiment.get_scene_cfg(),
            self.arena_env.task.get_scene_cfg(),
        )
        observation_cfg = combine_configclass_instances(
            "ObservationCfg",
            self.arena_env.scene.get_observation_cfg(),
            self.arena_env.embodiment.get_observation_cfg(),
            self.arena_env.task.get_observation_cfg(),
        )
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
            device_registry = DeviceRegistry()
            teleop_device_cfg = device_registry.get_teleop_device_cfg(
                self.arena_env.teleop_device, self.arena_env.embodiment
            )
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

        rewards_cfg = combine_configclass_instances(
            "RewardsCfg",
            self.arena_env.scene.get_rewards_cfg(),
            self.arena_env.embodiment.get_rewards_cfg(),
            self.arena_env.task.get_rewards_cfg(),
        )

        curriculum_cfg = combine_configclass_instances(
            "CurriculumCfg",
            self.arena_env.scene.get_curriculum_cfg(),
            self.arena_env.embodiment.get_curriculum_cfg(),
            self.arena_env.task.get_curriculum_cfg(),
        )

        commands_cfg = combine_configclass_instances(
            "CommandsCfg",
            self.arena_env.scene.get_commands_cfg(),
            self.arena_env.embodiment.get_commands_cfg(),
            self.arena_env.task.get_commands_cfg(),
        )

        isaaclab_arena_env = self.arena_env

        viewer_cfg = self.arena_env.task.get_viewer_cfg()

        episode_length_s = self.arena_env.task.get_episode_length_s()

        # Build the environment configuration
        if not self.args.mimic:
            env_cfg = IsaacLabArenaManagerBasedRLEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                rewards=rewards_cfg,
                curriculum=curriculum_cfg,
                commands=commands_cfg,
                xr=xr_cfg,
                teleop_devices=teleop_device_cfg,
                recorders=recorder_manager_cfg,
                metrics=metrics,
                isaaclab_arena_env=isaaclab_arena_env,
                viewer=viewer_cfg,
            )
            if episode_length_s is not None:
                env_cfg.episode_length_s = episode_length_s
        else:
            task_mimic_env_cfg = self.arena_env.task.get_mimic_env_cfg(arm_mode=self.arena_env.embodiment.arm_mode)
            env_cfg = IsaacArenaManagerBasedMimicEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                rewards=rewards_cfg,
                curriculum=curriculum_cfg,
                commands=commands_cfg,
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
                isaaclab_arena_env=isaaclab_arena_env,
                viewer=viewer_cfg,
            )

        # Apply the environment configuration callback if it is set
        # This can be used to modify the simulation configuration, etc.
        if self.arena_env.env_cfg_callback is not None:
            env_cfg = self.arena_env.env_cfg_callback(env_cfg)

        return env_cfg

    def get_entry_point(self) -> str | type[ManagerBasedRLMimicEnv]:
        """Return the entry point of the environment."""
        if self.args.mimic:
            return self.arena_env.embodiment.get_mimic_env()
        else:
            return "isaaclab.envs:ManagerBasedRLEnv"

    def build_registered(
        self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None
    ) -> tuple[str, IsaacLabArenaManagerBasedRLEnvCfg]:
        """Register Gym env and parse runtime cfg."""
        name = self.arena_env.name
        # orchestrate the environment member interaction
        self.orchestrate()
        cfg_entry = env_cfg if env_cfg is not None else self.compose_manager_cfg()
        # THIS IS A WORKAROUND TO ALLOW USER TO GRADUALLY MOVE TO THE NEW CONFIGURATION SYSTEM.
        # THIS WILL BE REMOVED IN THE FUTURE.
        cfg_entry = self.modify_env_cfg(cfg_entry)
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

    def make_registered(self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None) -> ManagerBasedEnv:
        env, _ = self.make_registered_and_return_cfg(env_cfg)
        return env

    def make_registered_and_return_cfg(
        self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None
    ) -> tuple[ManagerBasedEnv, IsaacLabArenaManagerBasedRLEnvCfg]:
        name, cfg = self.build_registered(env_cfg)
        return gym.make(name, cfg=cfg).unwrapped, cfg
