# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from isaaclab.devices.device_base import DeviceCfg, DevicesCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_teleop import IsaacTeleopCfg

from isaaclab_arena.assets.registries import DeviceRegistry
from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    ArenaPhysicsCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.progress_tracking.progress_tracker import (
    make_progress_tracking_events_cfg,
    make_progress_tracking_recorder_cfg,
)
from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.terms.events import ResetBackgroundPhysics
from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.variations import variations_hydra
from isaaclab_arena.variations.variation_base import VariationBase
from isaaclab_arena.variations.variation_recorder import VariationRecorder


class LegacySingleEmbodimentBuilder(ArenaEnvBuilder):
    """Snapshot of single-robot assembly before the embodiment list.

    Inherited helpers retain shared metadata and physics setup. The assembly
    method remains independent of the new loops and combination helpers.
    """

    def compose_manager_cfg(self) -> tuple[IsaacLabArenaManagerBasedRLEnvCfg, dict[str, Any]]:
        """Return the base ManagerBased cfg and the env kwargs (no registration).

        env_kwargs carries arguments to be forwarded to gym.make for construction of the IsaacLabArenaManagerBasedRLEnv.

        Returns:
            An (env_cfg, env_kwargs) tuple.
        """
        # Solve relations before building scene config so positions are captured correctly.
        if self.cfg.solve_relations:
            self._solve_relations()

        # Apply Hydra variation overrides. Needs to happen before build-time variations are applied.
        if self.hydra_overrides:
            variations: dict[str, list[VariationBase]] = self.get_all_variations()
            variations_hydra.apply_overrides(variations, self.hydra_overrides)

        # Attach the variation recorder before any sampling, so it observes both build-time samples
        # (drawn just below) and run-time samples (drawn during simulation).
        variation_recorder = VariationRecorder()
        variation_recorder.attach(self.get_all_variations())

        # Apply build-time variations now, before scene_cfg is materialised.
        self._apply_build_time_variations()

        resolved_physics_backend = self.resolved_physics_backend

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        embodiment = self.arena_env.embodiment or NoEmbodiment()
        embodiment.configure_physics_backend(resolved_physics_backend)
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
                [(PLACEMENT_RESET_EVENT_NAME, EventTermCfg, self._placement_event_cfg)],
            )
            placement_event_cfg = PlacementEventCfg()
        variations_event_cfg = self._compose_variations_event_cfg()
        progress_objectives = task.get_progress_objectives()
        progress_tracking_events_cfg: Any = (
            make_progress_tracking_events_cfg(progress_objectives) if progress_objectives else None
        )
        background_physics_events_cfg = None
        background_physics_paths = self.arena_env.scene.get_background_physics_paths()
        if background_physics_paths:
            reset_background_physics = EventTermCfg(
                func=ResetBackgroundPhysics,
                mode="reset",
                params={
                    "background_prim_paths": self.arena_env.scene.get_background_physics_prim_paths(),
                    "physics_paths": background_physics_paths,
                    "referenced_paths": self.arena_env.scene.get_background_physics_referenced_paths(),
                },
            )
            BackgroundPhysicsEventsCfg = make_configclass(
                "BackgroundPhysicsEventsCfg",
                [("reset_background_physics", EventTermCfg, reset_background_physics)],
            )
            background_physics_events_cfg = BackgroundPhysicsEventsCfg()
        # Keep the background term first so its one-time snapshot observes the
        # composed startup state before any reset event can mutate scene entities.
        events_cfg = combine_configclass_instances(
            "EventsCfg",
            background_physics_events_cfg,
            embodiment.get_events_cfg(),
            self.arena_env.scene.get_events_cfg(),
            task.get_events_cfg(),
            placement_event_cfg,
            variations_event_cfg,
            progress_tracking_events_cfg,
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
        metrics_cfg = self._compose_metrics_cfg(metrics)
        metrics_recorder_manager_cfg = metrics_to_recorder_manager_cfg(metrics)
        progress_tracking_recorder_cfg: Any = (
            make_progress_tracking_recorder_cfg(progress_objectives) if progress_objectives else None
        )

        # Base has to be specified explicitly to avoid type errors and not lose inheritance.
        recorder_manager_cfg = combine_configclass_instances(
            "RecorderManagerCfg",
            metrics_recorder_manager_cfg,
            task.get_recorder_term_cfg(),
            embodiment.get_recorder_term_cfg(record_trajectories=self.cfg.record_trajectories),
            progress_tracking_recorder_cfg,
            bases=(RecorderManagerBaseCfg,),
        )
        recorder_manager_cfg = self._modify_recorder_cfg_dataset_filename(recorder_manager_cfg)
        # Eval runs overwrite the timestamped default so rebuilds do not clobber each other.
        if self.cfg.recorder_dataset_filename is not None:
            recorder_manager_cfg.dataset_filename = self.cfg.recorder_dataset_filename
        if self.cfg.recorder_dataset_export_dir_path is not None:
            recorder_manager_cfg.dataset_export_dir_path = self.cfg.recorder_dataset_export_dir_path

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

        episode_recorders_cfg = self._compose_episode_recorders_cfg(self.arena_env.episode_recorder_terms)

        viewer_cfg = task.get_viewer_cfg()

        episode_length_s = task.get_episode_length_s()

        task_description = self.cfg.language_instruction or task.get_task_description()

        demo_recorder_config = ArenaEnvRecorderManagerCfg() if embodiment.enable_cameras else None

        # Build the environment configuration
        assert not self.cfg.mimic, "The reference fixture covers ordinary single-robot assembly"
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
            demo_recorder_config=demo_recorder_config,
            metrics=metrics_cfg,
            episode_recorders=episode_recorders_cfg,
            task_description=task_description,
            viewer=viewer_cfg,
        )
        # Tasks always resolve to a concrete episode length.
        env_cfg.episode_length_s = episode_length_s

        # Set seed for Isaac Lab env.
        env_cfg.seed = self.cfg.seed

        arena_physics = ArenaPhysicsCfg()
        if resolved_physics_backend is PhysicsBackend.PHYSX:
            env_cfg.sim.physics = arena_physics.physx
        elif resolved_physics_backend is PhysicsBackend.NEWTON:
            env_cfg.sim.physics = arena_physics.newton
            # replicate_physics=False is not supported for Newton, so force it to True here.
            # submodules/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:120
            env_cfg.scene.replicate_physics = True

        if self.arena_env.env_cfg_callback is not None:
            env_cfg = self.arena_env.env_cfg_callback(env_cfg)
            if resolved_physics_backend is PhysicsBackend.PHYSX:
                assert isinstance(
                    env_cfg.sim.physics, PhysxCfg
                ), "env_cfg_callback changed the physics backend away from PhysX."
            elif resolved_physics_backend is PhysicsBackend.NEWTON:
                assert isinstance(
                    env_cfg.sim.physics, NewtonCfg
                ), "env_cfg_callback changed the physics backend away from Newton."

        env_kwargs: dict[str, Any] = {"variation_recorder": variation_recorder}
        return env_cfg, env_kwargs
