# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import gymnasium as gym
from typing import Any

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
from isaaclab_arena.progress_tracking.progress_tracker import (
    make_progress_tracking_events_cfg,
    make_progress_tracking_recorder_cfg,
)
from isaaclab_arena.recording.common_terms import CoreEpisodeRecorderTermCfg, VariationEpisodeRecorderTermCfg
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
from isaaclab_arena.recording.progress_terms import ProgressEpisodeRecorderTermCfg
from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg
from isaaclab_arena.utils.multiprocess import get_local_rank
from isaaclab_arena.variations import variations_hydra, variations_printing
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBase
from isaaclab_arena.variations.variation_recorder import VariationRecorder


class ArenaEnvBuilder:
    """Compose IsaacLab Arena → IsaacLab configs"""

    def __init__(
        self,
        arena_env: IsaacLabArenaEnvironment,
        args: argparse.Namespace,
        hydra_overrides: list[str] | None = None,
    ):
        self.arena_env = arena_env
        self.args = args
        self.hydra_overrides = hydra_overrides
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
            random_yaw_init=self.args.random_yaw_init,
        )

    def get_all_variations(self) -> dict[str, list[VariationBase]]:
        """Return ``{asset_name: [variation, ...]}`` for every variation host in the env.

        Merges scene variations with the embodiment own variations.
        """
        scene_and_embodiment_variations = self.arena_env.scene.get_asset_variations()
        if self.arena_env.embodiment is not None:
            embodiment_variations = self.arena_env.embodiment.get_variations()
            scene_and_embodiment_variations[self.arena_env.embodiment.name] = embodiment_variations
        return scene_and_embodiment_variations

    def get_variations_catalogue_as_string(self) -> str:
        """Return a human-readable catalog of Hydra-configurable variations for this env."""
        variations: dict[str, list[VariationBase]] = self.get_all_variations()
        return variations_printing.get_variations_catalogue_as_string(variations, hydra_overrides=self.hydra_overrides)

    def _compose_variations_event_cfg(self) -> Any | None:
        """Build a configclass with one :class:`EventTermCfg` per enabled run-time variation.

        Returns ``None`` when no run-time variation is enabled.
        """
        # Assemble all the variations together into a single configclass.
        fields: list[tuple[str, type, EventTermCfg]] = []
        added_event_names: set[str] = set()
        for variations_per_asset in self.get_all_variations().values():
            for variation in variations_per_asset:
                if not variation.enabled:
                    continue
                if not isinstance(variation, RunTimeVariationBase):
                    continue
                event_name, event_cfg = variation.build_event_cfg()
                assert event_name not in added_event_names, (
                    f"Duplicate variation event term name '{event_name}'. "
                    "Each variation must produce a unique name; consider prefixing with the asset name."
                )
                added_event_names.add(event_name)
                fields.append((event_name, EventTermCfg, event_cfg))
        if not fields:
            return None
        VariationsEventCfg = make_configclass("VariationsEventCfg", fields)
        return VariationsEventCfg()

    def _apply_build_time_variations(self) -> None:
        """Sample and apply every enabled :class:`BuildTimeVariationBase`.

        These mutate asset configs in place (e.g. a dome light's spawner
        texture), so this must run before ``scene_cfg`` is materialised.
        """
        for asset_variations in self.get_all_variations().values():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                if not isinstance(variation, BuildTimeVariationBase):
                    continue
                variation.apply()

    def _modify_recorder_cfg_dataset_filename(self, recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
        """Modify the recorder dataset filename to include the timestamp and rank."""
        base = getattr(recorder_cfg, "dataset_filename", "dataset")
        recorder_cfg.dataset_filename = (
            f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{get_local_rank()}"
        )
        return recorder_cfg

    def _compose_metrics_cfg(self, metrics: list[MetricBase] | None) -> object | None:
        """Build a configclass container with one ``MetricTermCfg`` field per metric."""
        if not metrics:
            return None
        fields = [(m.name, MetricTermCfg, m.get_metric_term_cfg()) for m in metrics]
        return make_configclass("MetricsCfg", fields)()

    def _compose_episode_recorders_cfg(self, extra_terms: dict[str, EpisodeRecorderTermCfg] | None = None) -> object:
        """Build a configclass container with one EpisodeRecorderTermCfg field per episode recorder term.

        Note that this function automatically adds the core, variations, and progress terms. The
        progress term records nothing for tasks that define no progress objectives.
        """
        fields = [
            ("core", EpisodeRecorderTermCfg, CoreEpisodeRecorderTermCfg()),
            ("variations", EpisodeRecorderTermCfg, VariationEpisodeRecorderTermCfg()),
            ("progress", EpisodeRecorderTermCfg, ProgressEpisodeRecorderTermCfg()),
        ]
        for name, term_cfg in (extra_terms or {}).items():
            assert name not in (
                "core",
                "variations",
                "progress",
            ), f"Episode recorder term name '{name}' collides with a built-in term."
            fields.append((name, EpisodeRecorderTermCfg, term_cfg))
        return make_configclass("EpisodeRecorderManagerCfg", fields)()

    def compose_manager_cfg(self) -> tuple[IsaacLabArenaManagerBasedRLEnvCfg, dict[str, Any]]:
        """Return the base ManagerBased cfg and the env kwargs (no registration).

        env_kwargs carries arguments to be forwarded to gym.make for construction of the IsaacLabArenaManagerBasedRLEnv.

        Returns:
            An (env_cfg, env_kwargs) tuple.
        """
        # Solve relations before building scene config so positions are captured correctly.
        if self.args.solve_relations:
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
                [(PLACEMENT_RESET_EVENT_NAME, EventTermCfg, self._placement_event_cfg)],
            )
            placement_event_cfg = PlacementEventCfg()
        variations_event_cfg = self._compose_variations_event_cfg()
        progress_objectives = task.get_progress_objectives()
        progress_tracking_events_cfg: Any = (
            make_progress_tracking_events_cfg(progress_objectives) if progress_objectives else None
        )
        events_cfg = combine_configclass_instances(
            "EventsCfg",
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
            embodiment.get_recorder_term_cfg(),
            progress_tracking_recorder_cfg,
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

        episode_recorders_cfg = self._compose_episode_recorders_cfg(self.arena_env.episode_recorder_terms)

        viewer_cfg = task.get_viewer_cfg()

        episode_length_s = task.get_episode_length_s()

        # Language instruction is optionally overridden on the CLI.
        language_instruction = getattr(self.args, "language_instruction", None)
        task_description = language_instruction or task.get_task_description()

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
                episode_recorders=episode_recorders_cfg,
                task_description=task_description,
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
                task_description=task_description,
                viewer=viewer_cfg,
            )

        # Apply the environment configuration callback if it is set
        # This can be used to modify the simulation configuration, etc.
        if self.arena_env.env_cfg_callback is not None:
            env_cfg = self.arena_env.env_cfg_callback(env_cfg)

        # Set seed for Isaac Lab env.
        env_cfg.seed = self.args.seed

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

        env_kwargs: dict[str, Any] = {"variation_recorder": variation_recorder}
        return env_cfg, env_kwargs

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
        self,
        env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, IsaacLabArenaManagerBasedRLEnvCfg, dict[str, Any]]:
        """Build env cfg and register the env with gym. Stop short of env.make().

        The default operation is to call with no arguments, in which case the env_cfg is built from the
        Arena description passed to the builder at construction.

        Args:
            env_cfg: The optional environment cfg to use.
            env_kwargs: The optional environment kwargs to use.

        Returns:
            A ``(name, cfg, env_kwargs)`` tuple.
        """
        name = self.arena_env.name
        if env_cfg is None:
            env_cfg, env_kwargs = self.compose_manager_cfg()
        elif env_kwargs is None:
            env_kwargs = {}
        entry_point = self.get_entry_point()
        # Register the environment with the Gym registry.
        kwargs = {
            "env_cfg_entry_point": env_cfg,
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
        return name, cfg, env_kwargs

    def make_registered(
        self,
        env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None,
        env_kwargs: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> ManagerBasedEnv:
        """Build env cfg, register the env with gym, and make the env.

        The default operation is to call with no arguments, in which case the env_cfg is built from the
        Arena description passed to the builder at construction.

        Args:
            env_cfg: The optional environment cfg to use.
            env_kwargs: The optional environment kwargs to use.
            render_mode: The optional render mode to use.

        Returns:
            The environment.
        """
        env, _ = self.make_registered_and_return_cfg(env_cfg, env_kwargs, render_mode=render_mode)
        return env

    def make_registered_and_return_cfg(
        self,
        env_cfg: None | IsaacLabArenaManagerBasedRLEnvCfg = None,
        env_kwargs: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> tuple[ManagerBasedEnv, IsaacLabArenaManagerBasedRLEnvCfg]:
        """Build env cfg, register the env with gym, and make the env.

        The default operation is to call with no arguments, in which case the env_cfg is built from the
        Arena description passed to the builder at construction.

        Args:
            env_cfg: The optional environment cfg to use.
            env_kwargs: The optional environment kwargs to use.
            render_mode: The optional render mode to use.

        Returns:
            A tuple containing the environment and the environment configuration.
        """
        name, cfg, env_kwargs = self.build_registered(env_cfg, env_kwargs)
        env = gym.make(name, cfg=cfg, render_mode=render_mode, **env_kwargs)
        # ViewportCameraController sets the camera before KitVisualizer.initialize() is called,
        # so the call is silently ignored. Re-apply here once the visualizers are fully initialized.
        reapply_viewer_cfg(env)
        return env, cfg
