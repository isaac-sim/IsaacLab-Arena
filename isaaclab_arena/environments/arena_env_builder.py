# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import gymnasium as gym
from typing import TYPE_CHECKING, Any

from isaaclab.devices.device_base import DeviceCfg, DevicesCfg
from isaaclab.envs import ManagerBasedRLMimicEnv, ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.mdp import time_out
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_teleop import IsaacTeleopCfg

import isaaclab_arena_curobo  # noqa: F401
from isaaclab_arena.assets.registries import DeviceRegistry, PolicyRegistry
from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg, CameraResolutionMode
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    ArenaPhysicsCfg,
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.environments.relation_solver_interface import (
    make_placement_event,
    prepare_live_relation_placement,
    prepare_recorded_relation_placement,
)
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg
from isaaclab_arena.recording.common_terms import (
    CoreEpisodeRecorderTermCfg,
    PlacementEpisodeRecorderTermCfg,
    VariationEpisodeRecorderTermCfg,
)
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
from isaaclab_arena.recording.progress_terms import ProgressEpisodeRecorderTermCfg
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import PLACEMENT_RESET_EVENT_NAME
from isaaclab_arena.relations.placement_sampler import (
    PlacementSampler,
    RecordedPlacementSampler,
    serialize_placement_samples,
    supports_recorded_placement,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import task_success_from_progress
from isaaclab_arena.terms.events import ResetBackgroundPhysics
from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.isaaclab_utils.warp_patch import install_empty_cpu_warp_to_torch_patch
from isaaclab_arena.utils.multiprocess import get_local_rank
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.variations import variations_hydra, variations_printing
from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBase, get_enabled_variations_by_key
from isaaclab_arena.variations.variation_recorder import VariationRecorder
from isaaclab_arena.variations.variation_replay_scheduler import VariationReplayScheduler

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyCfg


# TODO: peterd, 2026-09-16: Remove this once Arena migrates off ViewCfg and into KitVisualizerCfg.
def _configure_arena_visualizer_defaults(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
    """Preserve Arena's native scene background and migrate its legacy task camera configuration.

    Compatibility layer to enable Arena's legacy ViewerCfg to work with Lab's default_visualizer_cfg.
    This can be removed once Arena migrates off ViewCfg and into KitVisualizerCfg.
    """
    from isaaclab.envs.common import _apply_deprecated_viewer_cfg

    if env_cfg.sim.default_visualizer_cfg is None:
        _apply_deprecated_viewer_cfg(env_cfg)
    visualizer_cfg = env_cfg.sim.default_visualizer_cfg
    if visualizer_cfg is None:
        visualizer_cfg = VisualizerCfg(background_color=None)
    elif visualizer_cfg.background_color == VisualizerCfg().background_color:
        visualizer_cfg.background_color = None
    env_cfg.sim.default_visualizer_cfg = visualizer_cfg
    # Avoid a second deprecated conversion when ManagerBasedRLEnv initializes.
    env_cfg.viewer = ViewerCfg()


class ArenaEnvBuilder:
    """Build Isaac Lab configurations and registered environments from an Arena environment."""

    def __init__(
        self,
        arena_env: IsaacLabArenaEnvironment,
        cfg: ArenaEnvBuilderCfg,
        hydra_overrides: list[str] | None = None,
        policy_cfg: PolicyCfg | None = None,
    ):
        """Create a builder for one Arena environment.

        Args:
            arena_env: The composed Arena environment to build.
            cfg: Builder options such as the environment count and camera resolution mode.
            hydra_overrides: Variation overrides applied when the environment config is composed.
            policy_cfg: Config of the registered policy that will drive the environment. Required by
                ``cfg.camera_resolution_mode`` ``policy_default`` to fit the embodiment cameras to the policy.
        """
        self.arena_env = arena_env
        self.cfg = cfg
        self.hydra_overrides = hydra_overrides
        self._apply_camera_resolution_mode(policy_cfg)
        self.interactive_scene_cfg = InteractiveSceneCfg(
            num_envs=cfg.num_envs, env_spacing=cfg.env_spacing, replicate_physics=False
        )

    def _apply_camera_resolution_mode(self, policy_cfg: PolicyCfg | None) -> None:
        """Resize the embodiment camera rig as selected by ``cfg.camera_resolution_mode``."""
        if self.cfg.camera_resolution_mode == CameraResolutionMode.EMBODIMENT_DEFAULT:
            return
        assert self.cfg.camera_resolution_mode == CameraResolutionMode.POLICY_DEFAULT
        embodiment = self.arena_env.embodiment
        assert embodiment is not None, "camera_resolution_mode 'policy_default' requires an environment embodiment"
        assert (
            embodiment.enable_cameras
        ), "camera_resolution_mode 'policy_default' requires embodiment cameras to be enabled"
        assert (
            embodiment.camera_config is not None
        ), "camera_resolution_mode 'policy_default' requires an embodiment camera rig"
        assert policy_cfg is not None, "camera_resolution_mode 'policy_default' requires the policy config"
        policy_type = PolicyRegistry().get_policy_type_for_cfg(policy_cfg)
        policy_image_size = policy_type.default_image_size(policy_cfg)
        assert policy_image_size is not None, (
            "camera_resolution_mode 'policy_default' requires a policy that declares a default image size"
            f" (PolicyBase.default_image_size), but {policy_type.__name__} declares none; select"
            " 'embodiment_default' instead"
        )
        embodiment.camera_config.fit_resolution(*policy_image_size)

    @property
    def resolved_physics_backend(self) -> PhysicsBackend:
        """Return the physics backend selected for this build (CLI preset or environment default)."""
        return self.cfg.presets if self.cfg.presets is not None else self.arena_env.default_physics_backend

    def _prepare_relation_placement(
        self,
        scheduler: VariationReplayScheduler | None,
    ) -> PlacementSampler | RecordedPlacementSampler | None:
        """Prepare live or recorded relation placement.

        Recorded placement validates and binds native samples without constructing
        a solver pool. Live placement collects constrained assets and solves a pool.

        Live reset behaviour depends on ``ObjectPlacerParams.resolve_on_reset``.
        When the environment does not provide placer parameters, the builder creates
        them from ``ArenaEnvBuilderCfg``.

        * **True** (default) — registers a reset event that draws a fresh layout
          from the pool for each resetting environment.
        * **False** — assigns one fixed layout per environment and reuses it.

        Args:
            scheduler: Shared recorded-variation scheduler, if replay is enabled.

        Returns:
            The prepared live or recorded placement sampler, or ``None`` when placement is disabled.
        """
        if scheduler is not None and scheduler.variation_record.has_placement_samples:
            return prepare_recorded_relation_placement(
                replay_assets=self.arena_env.get_placement_assets(),
                scheduler=scheduler,
                num_envs=self.cfg.num_envs,
            )

        if not self.cfg.solve_relations:
            return None
        # Reachability constraints are defined in the task, so apply them before placement.
        if self.arena_env.task is not None:
            self.arena_env.task.apply_reachability_constraints()
        placement_assets = self.arena_env.scene.get_objects_with_relations()
        embodiment = self.arena_env.embodiment
        if embodiment is not None and embodiment.get_relations():
            placement_assets.append(embodiment)

        placer_params = self.arena_env.placer_params
        if placer_params is None:
            placer_params = ObjectPlacerParams(
                solver_params=RelationSolverParams(verbose=False, save_position_history=False),
            )
        if self.cfg.placement_seed is not None:
            placer_params.placement_seed = self.cfg.placement_seed
        if self.cfg.resolve_on_reset is not None:
            placer_params.resolve_on_reset = self.cfg.resolve_on_reset

        # Delists itself unless the embodiment has a registered cuRobo config and the solver deps are importable.
        # TODO(xinjieyao, 2026-07-22): updated once robot-object co-placement is merged.
        placer_params.reachability_config.embodiment = self.arena_env.embodiment
        return prepare_live_relation_placement(
            placement_assets,
            num_envs=self.cfg.num_envs,
            placer_params=placer_params,
            scene_assets=self.arena_env.scene.assets.values(),
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

    def _build_variation_events_cfg(self) -> Any | None:
        """Build a configclass with one EventTermCfg per enabled run-time variation.

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
        """Configure every enabled variation at build time before ``scene_cfg`` is materialised.

        These mutate asset configs in place (e.g. a dome light's spawner
        texture), so this must run before ``scene_cfg`` is materialised.
        """
        for variation_key, variation in get_enabled_variations_by_key(self.get_all_variations()).items():
            variation.configure_at_build_time(seed=self.cfg.seed, variation_key=variation_key)

    def _modify_recorder_cfg_dataset_filename(self, recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
        """Modify the recorder dataset filename to include the timestamp and rank."""
        base = getattr(recorder_cfg, "dataset_filename", "dataset")
        recorder_cfg.dataset_filename = (
            f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{get_local_rank()}"
        )
        return recorder_cfg

    def _build_metrics_cfg(self, metrics: list[MetricBase] | None) -> object | None:
        """Build a configclass container with one ``MetricTermCfg`` field per metric."""
        if not metrics:
            return None
        fields = [(m.name, MetricTermCfg, m.get_metric_term_cfg()) for m in metrics]
        return make_configclass("MetricsCfg", fields)()

    def _build_termination_manager_cfg(
        self,
        task_termination_cfg: TaskTerminationCfg,
    ) -> object:
        """Translate TaskTerminationCfg into an Isaac Lab termination configclass.

        Args:
            task_termination_cfg: Task-owned success criteria, failures, and timeout.

        Returns:
            A configclass containing the requested failure, timeout, and success terms.
        """
        termination_terms = dict(task_termination_cfg.failures)
        if task_termination_cfg.timeout_s is not None:
            termination_terms["time_out"] = TerminationTermCfg(func=time_out, time_out=True)
        success_criteria = task_termination_cfg.success

        # Install the shared success term when the task defines success criteria.
        if success_criteria:
            success_term = TerminationTermCfg(func=task_success_from_progress)
            termination_terms["success"] = success_term
        termination_fields = [(name, TerminationTermCfg, term) for name, term in termination_terms.items()]
        return make_configclass("TerminationsCfg", termination_fields)()

    def _build_episode_recorders_cfg(self, extra_terms: dict[str, EpisodeRecorderTermCfg] | None = None) -> object:
        """Build a configclass container with one EpisodeRecorderTermCfg field per episode recorder term.

        Includes the built-in core, placement, variations, and progress terms. The
        progress term records nothing for tasks that define no completion criteria.
        """
        fields = [
            ("core", EpisodeRecorderTermCfg, CoreEpisodeRecorderTermCfg()),
            ("placement", EpisodeRecorderTermCfg, PlacementEpisodeRecorderTermCfg()),
            ("variations", EpisodeRecorderTermCfg, VariationEpisodeRecorderTermCfg()),
            ("progress", EpisodeRecorderTermCfg, ProgressEpisodeRecorderTermCfg()),
        ]
        for name, term_cfg in (extra_terms or {}).items():
            assert name not in (
                "core",
                "placement",
                "variations",
                "progress",
            ), f"Episode recorder term name '{name}' collides with a built-in term."
            fields.append((name, EpisodeRecorderTermCfg, term_cfg))
        return make_configclass("EpisodeRecorderManagerCfg", fields)()

    def build_env_cfg(self) -> tuple[IsaacLabArenaManagerBasedRLEnvCfg, dict[str, Any]]:
        """Build the environment configuration and constructor arguments without registering it.

        Applies variation overrides and build-time variations, prepares relation placement,
        and combines scene, embodiment, and task settings. Updates the components of ``arena_env`` in place.

        Returns:
            An (env_cfg, env_kwargs) tuple, with constructor arguments to forward to gym.make.
        """
        # Apply Hydra variation overrides. Needs to happen before build-time variations are applied.
        variations = self.get_all_variations()
        if self.hydra_overrides:
            variations_hydra.apply_overrides(variations, self.hydra_overrides)

        variation_replay_scheduler = None
        if self.cfg.recorded_variation_samples_path is not None:
            variation_replay_scheduler = configure_recorded_variation_replay(
                self.cfg.recorded_variation_samples_path,
                variations,
            )

        # Attach the variation recorder before any sampling, so it observes both build-time samples
        # (drawn just below) and run-time samples (drawn during simulation).
        variation_recorder = VariationRecorder()
        variation_recorder.attach(variations)

        # Apply build-time variations before self.arena_env.scene.get_scene_cfg() copies each object's object_cfg
        # into scene_cfg below. They can change object geometry, so relation placement runs after them.
        self._apply_build_time_variations()

        # Placement needs its source selected before scene config captures construction poses.
        placement_sampler = self._prepare_relation_placement(variation_replay_scheduler)
        # A RigidObjectSet's selected member identity is absent from pose-only samples, so skip unreplayable records.
        if placement_sampler is not None and supports_recorded_placement(self.arena_env.get_placement_assets()):
            variation_recorder.attach_placement_sampler(
                placement_sampler,
                serialize_placement_samples,
            )

        resolved_physics_backend = self.resolved_physics_backend

        # Constructing the environment by combining inputs from the scene, embodiment, and task.
        embodiment = self.arena_env.embodiment or NoEmbodiment()
        embodiment.configure_physics_backend(resolved_physics_backend)
        task = self.arena_env.task or NoTask()
        task.configure_for_embodiment(embodiment)
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
        if placement_sampler is not None:
            PlacementEventCfg = make_configclass(
                "PlacementEventCfg",
                [(PLACEMENT_RESET_EVENT_NAME, EventTermCfg, make_placement_event(placement_sampler))],
            )
            placement_event_cfg = PlacementEventCfg()
        variations_event_cfg = self._build_variation_events_cfg()
        task_termination_cfg = task.get_termination_cfg()
        assert isinstance(
            task_termination_cfg, TaskTerminationCfg
        ), "Tasks must return TaskTerminationCfg with success criteria, failures, and timeout_s."
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
        )
        termination_cfg = self._build_termination_manager_cfg(task_termination_cfg)
        task_progress_cfg = None
        tracked_events = task.get_tracked_events()
        assert not tracked_events or task_termination_cfg.success, "Tracked events require task success criteria."
        if task_termination_cfg.success:
            task_progress_cfg = TaskProgressCfg(
                success_criteria=task_termination_cfg.success,
                subtasks_are_sequential=task_termination_cfg.subtasks_are_sequential,
                desired_subtask_success_state=task_termination_cfg.desired_subtask_success_state,
                tracked_events=tracked_events,
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
        metrics_cfg = self._build_metrics_cfg(metrics)
        metrics_recorder_manager_cfg = metrics_to_recorder_manager_cfg(metrics)
        # Base has to be specified explicitly to avoid type errors and not lose inheritance.
        recorder_manager_cfg = combine_configclass_instances(
            "RecorderManagerCfg",
            metrics_recorder_manager_cfg,
            task.get_recorder_term_cfg(),
            embodiment.get_recorder_term_cfg(record_trajectories=self.cfg.record_trajectories),
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

        episode_recorders_cfg = self._build_episode_recorders_cfg(self.arena_env.episode_recorder_terms)

        viewer_cfg = task.get_viewer_cfg()

        episode_length_s = task_termination_cfg.timeout_s
        if episode_length_s is None:
            # Isaac Lab still needs a numeric episode length without timeout termination.
            episode_length_s = task.get_episode_length_s()

        task_description = self.cfg.language_instruction or task.get_task_description()

        demo_recorder_config = ArenaEnvRecorderManagerCfg() if embodiment.enable_cameras else None

        # Build the environment configuration
        if not self.cfg.mimic:
            env_cfg = IsaacLabArenaManagerBasedRLEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                task_progress=task_progress_cfg,
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
        else:
            assert not isinstance(embodiment, NoEmbodiment), "Mimic mode requires an embodiment to be specified"
            assert not isinstance(task, NoTask), "Mimic mode requires a task to be specified"
            task_mimic_env_cfg = task.get_mimic_env_cfg(arm_mode=self.arena_env.embodiment.arm_mode)
            mimic_recorder_config = task_mimic_env_cfg.mimic_recorder_config
            if mimic_recorder_config is None:
                mimic_recorder_config = demo_recorder_config
            env_cfg = IsaacArenaManagerBasedMimicEnvCfg(
                observations=observation_cfg,
                actions=actions_cfg,
                events=events_cfg,
                scene=scene_cfg,
                terminations=termination_cfg,
                task_progress=task_progress_cfg,
                rewards=rewards_cfg,
                curriculum=curriculum_cfg,
                commands=commands_cfg,
                xr=xr_cfg,
                isaac_teleop=isaac_teleop_cfg,
                teleop_devices=teleop_devices_cfg,
                demo_recorder_config=demo_recorder_config,
                # Mimic stuff
                datagen_config=task_mimic_env_cfg.datagen_config,
                subtask_configs=task_mimic_env_cfg.subtask_configs,
                task_constraint_configs=task_mimic_env_cfg.task_constraint_configs,
                mimic_recorder_config=mimic_recorder_config,
                # NOTE(alexmillane, 2025-09-25): Metric + recorders excluded from mimic env,
                # I assume that they're not needed for the mimic env.
                # recorders=recorder_manager_cfg,
                # metrics=metrics_cfg,
                task_description=task_description,
                viewer=viewer_cfg,
            )

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

        env_kwargs: dict[str, Any] = {
            "variation_recorder": variation_recorder,
            "variation_replay_scheduler": variation_replay_scheduler,
        }
        return env_cfg, env_kwargs

    def get_entry_point(self) -> str | type[ManagerBasedRLMimicEnv]:
        """Return the entry point of the environment."""
        if self.cfg.mimic:
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
        The registered constructor supplies env_kwargs even when the caller only passes cfg to gym.make.
        A registration shares samplers across constructions; close its current env before making another.
        Each construction starts fresh runtime recording and restarts recorded replay.

        Args:
            env_cfg: The optional environment cfg to use.
            env_kwargs: The optional environment kwargs to use.

        Returns:
            A ``(name, cfg, env_kwargs)`` tuple.
        """
        install_empty_cpu_warp_to_torch_patch()
        apply_arena_global_settings()
        name = self.arena_env.name
        if env_cfg is None:
            env_cfg, env_kwargs = self.build_env_cfg()
        elif env_kwargs is None:
            env_kwargs = {}
        _configure_arena_visualizer_defaults(env_cfg)
        self.arena_env.scene.validate_simulation_cfg(env_cfg.sim)
        entry_point = self.get_entry_point()
        # The registered factory retains runtime objects for Gym-only callers.
        # VariationRecorder holds the env weakly, so the recorder does not keep a closed simulation alive.

        def create_env(**kwargs):
            # Sampler callbacks capture these objects, so preserve their identity across Gym's kwargs deepcopy.
            env_type = (
                gym.envs.registration.load_env_creator(entry_point) if isinstance(entry_point, str) else entry_point
            )
            return env_type(**(env_kwargs | kwargs))

        kwargs = {
            "env_cfg_entry_point": env_cfg,
        }
        if env_cfg.demo_recorder_config is not None:
            kwargs["demo_recorder_cfg_entry_point"] = env_cfg.demo_recorder_config
        if self.arena_env.rl_framework_entry_point is not None:
            kwargs[self.arena_env.rl_framework_entry_point] = self.arena_env.rl_policy_cfg
        gym.register(
            id=name,
            entry_point=create_env,
            kwargs=kwargs,
            disable_env_checker=True,
        )
        cfg = parse_env_cfg(
            name,
            device=self.cfg.device,
            num_envs=self.cfg.num_envs,
            use_fabric=not self.cfg.disable_fabric,
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
        name, cfg, _ = self.build_registered(env_cfg, env_kwargs)
        env = gym.make(name, cfg=cfg, render_mode=render_mode)
        return env, cfg
