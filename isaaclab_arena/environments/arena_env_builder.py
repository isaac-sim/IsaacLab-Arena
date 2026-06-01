# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import gymnasium as gym

from isaaclab.devices.device_base import DeviceCfg, DevicesCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_teleop import IsaacTeleopCfg

from isaaclab_arena.assets.registries import DeviceRegistry
from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg
from isaaclab_arena.utils.multiprocess import get_local_rank


class ArenaEnvBuilder:
    """Compose IsaacLab Arena → IsaacLab configs"""

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args
        self.interactive_scene_cfg = InteractiveSceneCfg(
            num_envs=args.num_envs, env_spacing=args.env_spacing, replicate_physics=False
        )
        self._placement_event_cfg: EventTermCfg | None = None

    def _solve_relations(self) -> None:
        """Solve spatial relations for objects in the scene.

        This method:
        1. Collects all objects from the scene that have relations
        2. Builds an object-placement pool
        3. Reuses the object-only relation placer
        4. Applies solved positions either by writing fixed per-object initial poses
           or by registering a pooled reset placement event

        Behaviour on reset depends on :attr:`ObjectPlacerParams.resolve_on_reset`
        (overridable from CLI with --resolve_on_reset / --no-resolve_on_reset):

        * **True** (default) — registers a reset event that draws a fresh layout
          from the pool for each resetting environment.
        * **False** — applies one layout per environment so per-object reset
          events restore the same layout every time.
        """
        objects_with_relations = self.arena_env.scene.get_objects_with_relations()
        self._placement_event_cfg = solve_and_apply_relation_placement(
            objects_with_relations,
            num_envs=self.args.num_envs,
            placement_seed=self.args.placement_seed,
            resolve_on_reset=self.args.resolve_on_reset,
        )

    def _modify_recorder_cfg_dataset_filename(self, recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
        """Modify the recorder dataset filename to include the timestamp and rank."""
        base = getattr(recorder_cfg, "dataset_filename", "dataset")
        recorder_cfg.dataset_filename = (
            f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{get_local_rank()}"
        )
        return recorder_cfg

    @staticmethod
    def _metrics_to_metrics_cfg(metrics: list[MetricBase] | None) -> object | None:
        """Build a configclass container with one ``MetricTermCfg`` field per metric."""
        if not metrics:
            return None
        fields = [(m.name, MetricTermCfg, m.get_metric_term_cfg()) for m in metrics]
        return make_configclass("MetricsCfg", fields)()

    def compose_manager_cfg(self) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Return base ManagerBased cfg (scene+events+terminations+xr), no registration."""

        # Solve relations before building scene config so positions are captured correctly.
        if self.args.solve_relations:
            self._solve_relations()

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        embodiment = self.arena_env.embodiment or NoEmbodiment()
        task = self.arena_env.task or NoTask()
        scene_cfg = combine_configclass_instances(
            "SceneCfg",
            self.interactive_scene_cfg,
            self.arena_env.scene.get_scene_cfg(),
            embodiment.get_scene_cfg(),
            task.get_scene_cfg(),
        )
        observation_cfg = combine_configclass_instances(
            "ObservationCfg",
            self.arena_env.scene.get_observation_cfg(),
            embodiment.get_observation_cfg(),
            task.get_observation_cfg(),
        )
        placement_event_cfg = None
        if self._placement_event_cfg is not None:
            PlacementEventCfg = make_configclass(
                "PlacementEventCfg",
                [("placement_reset", EventTermCfg, self._placement_event_cfg)],
            )
            placement_event_cfg = PlacementEventCfg()
        events_cfg = combine_configclass_instances(
            "EventsCfg",
            embodiment.get_events_cfg(),
            self.arena_env.scene.get_events_cfg(),
            task.get_events_cfg(),
            placement_event_cfg,
        )
        termination_cfg = combine_configclass_instances(
            "TerminationCfg",
            task.get_termination_cfg(),
            self.arena_env.scene.get_termination_cfg(),
            embodiment.get_termination_cfg(),
        )
        actions_cfg = embodiment.get_action_cfg()
        xr_cfg = embodiment.get_xr_cfg()
        isaac_teleop_cfg = None
        teleop_devices_cfg = None
        if self.arena_env.teleop_device is not None:
            device_registry = DeviceRegistry()
            device_cfg = device_registry.get_teleop_device_cfg(self.arena_env.teleop_device, self.arena_env.embodiment)
            if isinstance(device_cfg, IsaacTeleopCfg):
                isaac_teleop_cfg = device_cfg
            elif isinstance(device_cfg, DeviceCfg):
                teleop_devices_cfg = DevicesCfg(devices={self.arena_env.teleop_device.name: device_cfg})
        metrics = task.get_metrics()
        metrics_cfg = self._metrics_to_metrics_cfg(metrics)
        metrics_recorder_manager_cfg = metrics_to_recorder_manager_cfg(metrics)

        # Base has to be specified explicitly to avoid type errors and not lose inheritance.
        recorder_manager_cfg = combine_configclass_instances(
            "RecorderManagerCfg",
            metrics_recorder_manager_cfg,
            task.get_recorder_term_cfg(),
            embodiment.get_recorder_term_cfg(),
            bases=(RecorderManagerBaseCfg,),
        )
        recorder_manager_cfg = self._modify_recorder_cfg_dataset_filename(recorder_manager_cfg)

        rewards_cfg = combine_configclass_instances(
            "RewardsCfg",
            self.arena_env.scene.get_rewards_cfg(),
            embodiment.get_rewards_cfg(),
            task.get_rewards_cfg(),
        )

        curriculum_cfg = combine_configclass_instances(
            "CurriculumCfg",
            self.arena_env.scene.get_curriculum_cfg(),
            embodiment.get_curriculum_cfg(),
            task.get_curriculum_cfg(),
        )

        commands_cfg = combine_configclass_instances(
            "CommandsCfg",
            self.arena_env.scene.get_commands_cfg(),
            embodiment.get_commands_cfg(),
            task.get_commands_cfg(),
        )

        isaaclab_arena_env = self.arena_env

        viewer_cfg = task.get_viewer_cfg()

        episode_length_s = task.get_episode_length_s()

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
                isaac_teleop=isaac_teleop_cfg,
                teleop_devices=teleop_devices_cfg,
                recorders=recorder_manager_cfg,
                metrics=metrics_cfg,
                isaaclab_arena_env=isaaclab_arena_env,
                viewer=viewer_cfg,
            )
            if episode_length_s is not None:
                env_cfg.episode_length_s = episode_length_s
        else:
            assert not isinstance(embodiment, NoEmbodiment), "Mimic mode requires an embodiment to be specified"
            assert not isinstance(task, NoTask), "Mimic mode requires a task to be specified"
            task_mimic_env_cfg = task.get_mimic_env_cfg(arm_mode=self.arena_env.embodiment.arm_mode)
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
                isaac_teleop=isaac_teleop_cfg,
                teleop_devices=teleop_devices_cfg,
                # Mimic stuff
                datagen_config=task_mimic_env_cfg.datagen_config,
                subtask_configs=task_mimic_env_cfg.subtask_configs,
                task_constraint_configs=task_mimic_env_cfg.task_constraint_configs,
                mimic_recorder_config=task_mimic_env_cfg.mimic_recorder_config,
                # NOTE(alexmillane, 2025-09-25): Metric + recorders excluded from mimic env,
                # I assume that they're not needed for the mimic env.
                # recorders=recorder_manager_cfg,
                # metrics=metrics_cfg,
                isaaclab_arena_env=isaaclab_arena_env,
                viewer=viewer_cfg,
            )

        # Apply the environment configuration callback if it is set
        # This can be used to modify the simulation configuration, etc.
        if self.arena_env.env_cfg_callback is not None:
            env_cfg = self.arena_env.env_cfg_callback(env_cfg)

        # Apply the --presets CLI flag (e.g. --presets newton).
        # This runs after the callback so the user's CLI choice is the final authority.
        presets = getattr(self.args, "presets", None)
        if presets is not None:
            from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

            env_cfg.sim.physics = getattr(ArenaPhysicsCfg(), presets)

            # Set replicate_physics for shared physics representations.
            # For Newton, without this flag, the simulation initialization
            # takes a very long time for large number of parallel environments.
            if presets == "newton":
                env_cfg.scene.replicate_physics = True

        return env_cfg

    def get_entry_point(self) -> str | type[ManagerBasedRLMimicEnv]:
        """Return the entry point of the environment."""
        if self.args.mimic:
            embodiment = self.arena_env.embodiment
            assert embodiment is not None and not isinstance(
                embodiment, NoEmbodiment
            ), "Mimic mode requires an embodiment to be specified"
            return embodiment.get_mimic_env()
        else:
            return "isaaclab_arena.environments.isaaclab_arena_manager_based_env:IsaacLabArenaManagerBasedRLEnv"

    def build_registered(
        self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None
    ) -> tuple[str, IsaacLabArenaManagerBasedRLEnvCfg]:
        """Register Gym env and parse runtime cfg."""
        name = self.arena_env.name
        cfg_entry = env_cfg if env_cfg is not None else self.compose_manager_cfg()
        entry_point = self.get_entry_point()
        # Register the environment with the Gym registry.
        kwargs = {
            "env_cfg_entry_point": cfg_entry,
        }
        if self.arena_env.rl_framework_entry_point is not None:
            kwargs[self.arena_env.rl_framework_entry_point] = self.arena_env.rl_policy_cfg
        gym.register(
            id=name,
            entry_point=entry_point,
            kwargs=kwargs,
            disable_env_checker=True,
        )
        cfg = parse_env_cfg(
            name,
            device=self.args.device,
            num_envs=self.args.num_envs,
            use_fabric=not self.args.disable_fabric,
        )
        return name, cfg

    def make_registered(
        self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None, render_mode: str | None = None
    ) -> ManagerBasedEnv:
        env, _ = self.make_registered_and_return_cfg(env_cfg, render_mode=render_mode)
        return env

    def make_registered_and_return_cfg(
        self, env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None, render_mode: str | None = None
    ) -> tuple[ManagerBasedEnv, IsaacLabArenaManagerBasedRLEnvCfg]:
        name, cfg = self.build_registered(env_cfg)
        env = gym.make(name, cfg=cfg, render_mode=render_mode)
        # ViewportCameraController sets the camera before KitVisualizer.initialize() is called,
        # so the call is silently ignored. Re-apply here once the visualizers are fully initialized.
        reapply_viewer_cfg(env)
        return env, cfg
