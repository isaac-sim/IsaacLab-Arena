# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from dataclasses import dataclass, fields, make_dataclass
from typing import Any, List, Tuple

class ArenaEnvBuilder:
    """Compose IsaacLab Arena → IsaacLab configs"""

    DEFAULT_SCENE_CFG = InteractiveSceneCfg(num_envs=4096, env_spacing=30.0, replicate_physics=False)

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args

    def orchestrate(self) -> None:
        """Orchestrate the environment member interaction"""
        if self.arena_env.orchestrator is not None:
            self.arena_env.orchestrator.orchestrate(
                self.arena_env.embodiment, self.arena_env.scene, self.arena_env.task
            )
    def add_object_collections(self, scene_cfg: InteractiveSceneCfg) -> InteractiveSceneCfg:
        """Add object collections to the scene configuration."""

        # add object_collection to scene_cfg

        objects_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=[
                    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd',
                    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd',
                    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd',
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
                ],
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            # spawn=sim_utils.MultiAssetSpawnerCfg(
            #     assets_cfg=[
            #         sim_utils.UsdFileCfg(
            #             usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd',
            #             scale=1,
            #             activate_contact_sensors=True,
            #         ),
            #         sim_utils.UsdFileCfg(
            #             usd_path='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd',
            #             scale=1,
            #             activate_contact_sensors=True,
            #         ),
            #     ],
            #     random_choice=True,
            #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #         solver_position_iteration_count=4, solver_velocity_iteration_count=0
            #     ),
            #     mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            #     collision_props=sim_utils.CollisionPropertiesCfg(),
            # ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1)),
        )

        # object_collection_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        #     rigid_objects={
        #         "object_A": RigidObjectCfg(
        #             prim_path="/World/envs/env_.*/Object_A",
        #             spawn=sim_utils.SphereCfg(
        #                 radius=0.1,
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
        #                 ),
        #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #                 collision_props=sim_utils.CollisionPropertiesCfg(),
        #             ),
        #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
        #         ),
        #         "object_B": RigidObjectCfg(
        #             prim_path="/World/envs/env_.*/Object_B",
        #             spawn=sim_utils.CuboidCfg(
        #                 size=(0.1, 0.1, 0.1),
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
        #                 ),
        #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #                 collision_props=sim_utils.CollisionPropertiesCfg(),
        #             ),
        #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 2.0)),
        #         ),
        #         "object_C": RigidObjectCfg(
        #             prim_path="/World/envs/env_.*/Object_C",
        #             spawn=sim_utils.ConeCfg(
        #                 radius=0.1,
        #                 height=0.3,
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #                     solver_position_iteration_count=4, solver_velocity_iteration_count=0
        #                 ),
        #                 mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #                 collision_props=sim_utils.CollisionPropertiesCfg(),
        #             ),
        #             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 2.0)),
        #         ),
            # }
        # )
        
        # Add new fields directly to the instance
        setattr(scene_cfg, "object", objects_cfg)
        # setattr(scene_cfg, "object_collection", object_collection_cfg)
        scene_cfg.env_spacing = 2.0
        
        return scene_cfg
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

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        scene_cfg = combine_configclass_instances(
            "SceneCfg",
            self.DEFAULT_SCENE_CFG,
            self.arena_env.scene.get_scene_cfg(),
            self.arena_env.embodiment.get_scene_cfg(),
            self.arena_env.task.get_scene_cfg(),
        )
        scene_cfg = self.add_object_collections(scene_cfg)
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
            task_mimic_env_cfg = self.arena_env.task.get_mimic_env_cfg(
                arm_mode=self.arena_env.embodiment.mimic_arm_mode
            )
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
