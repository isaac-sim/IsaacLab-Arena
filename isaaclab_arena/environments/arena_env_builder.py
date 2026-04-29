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

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.registries import DeviceRegistry
from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacArenaManagerBasedMimicEnvCfg,
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.metrics.recorder_manager_utils import metrics_to_recorder_manager_cfg
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_rotation_xyzw, solve_and_place_objects
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg
from isaaclab_arena.utils.multiprocess import get_local_rank
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class ArenaEnvBuilder:
    """Compose IsaacLab Arena → IsaacLab configs"""

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args
        self.interactive_scene_cfg = InteractiveSceneCfg(
            num_envs=args.num_envs, env_spacing=args.env_spacing, replicate_physics=False
        )
        self._placement_event_cfg: EventTermCfg | None = None

    def orchestrate(self) -> None:
        """Orchestrate the environment member interaction"""
        if self.arena_env.orchestrator is not None:
            self.arena_env.orchestrator.orchestrate(
                self.arena_env.embodiment, self.arena_env.scene, self.arena_env.task
            )

    def _solve_relations(self) -> None:
        """Solve spatial relations for objects in the scene.

        This method:
        1. Collects all objects from the scene that have relations
        2. Runs the ObjectPlacer to solve spatial constraints (no-overlap is built into the solver)
        3. Applies solved positions to objects via a :class:`PooledObjectPlacer`

        Behaviour on reset depends on :attr:`ObjectPlacerParams.resolve_on_reset`
        (overridable from CLI with ``--resolve_on_reset`` / ``--no-resolve_on_reset``):

        * **True** (default) — registers a reset event that draws a fresh layout
          from the pool for each resetting environment.
        * **False** — applies one layout per environment via ``set_initial_pose``
          so per-object reset events restore the same layout every time.
        """
        objects_with_relations = self.arena_env.scene.get_objects_with_relations()

        if not objects_with_relations:
            print("No objects with relations found in scene. Skipping relation solving.")
            return

        num_envs = self.args.num_envs
        cli_resolve = self.args.resolve_on_reset

        # The pool applies positions itself, so disable ObjectPlacer's built-in apply.
        # Position history and verbose logging are unnecessary for batch-solving a pool.
        placer_params = ObjectPlacerParams(
            placement_seed=self.args.placement_seed,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(save_position_history=False, verbose=False),
        )
        if cli_resolve is not None:
            placer_params.resolve_on_reset = cli_resolve

        pool_size = num_envs * placer_params.min_unique_layouts_per_env

        placement_pool = PooledObjectPlacer(
            objects=objects_with_relations,
            placer_params=placer_params,
            pool_size=pool_size,
        )

        if placer_params.resolve_on_reset:
            anchor_objects_set = set(get_anchor_objects(objects_with_relations))
            for obj in objects_with_relations:
                if obj not in anchor_objects_set and obj.event_cfg is not None:
                    raise RuntimeError(
                        f"Non-anchor object '{obj.name}' has an explicit pose-reset event. "
                        "Relational solving should not be combined with explicit setting of "
                        "poses on non-anchor objects."
                    )
            # Set init_state so objects spawn at valid positions (not origin).
            # The placement event will override these on every reset.
            self._set_init_state_from_pool(objects_with_relations, placement_pool, anchor_objects_set)
            self._placement_event_cfg = EventTermCfg(
                func=solve_and_place_objects,
                mode="reset",
                params={
                    "objects": objects_with_relations,
                    "placement_pool": placement_pool,
                },
            )
        else:
            self._apply_pool_layouts_to_objects(objects_with_relations, placement_pool, num_envs)

    def _set_init_state_from_pool(
        self,
        objects: list[Object | ObjectReference],
        pool: PooledObjectPlacer,
        anchor_objects_set: set,
    ) -> None:
        """Set ``object_cfg.init_state`` from a pool layout so objects spawn at valid positions.

        Only touches ``init_state.pos`` / ``init_state.rot`` — does NOT create
        per-object reset events (the placement event handles resets).
        """
        layout = pool.sample_with_replacement(1)[0]
        for obj in objects:
            if obj in anchor_objects_set:
                continue
            pos = layout.positions.get(obj)
            if pos is None:
                continue
            rotation_xyzw = get_rotation_xyzw(obj)
            obj.object_cfg.init_state.pos = pos
            obj.object_cfg.init_state.rot = rotation_xyzw

    def _apply_pool_layouts_to_objects(
        self,
        objects: list[Object | ObjectReference],
        pool: PooledObjectPlacer,
        num_envs: int,
    ) -> None:
        """Draw layouts from the pool and apply them to objects via ``set_initial_pose``.

        Each non-anchor object gets a :class:`~isaaclab_arena.utils.pose.PosePerEnv`
        so that per-object reset events restore these positions.
        """
        layouts = pool.sample_with_replacement(num_envs)
        anchor_objects_set = set(get_anchor_objects(objects))

        for obj in objects:
            if obj in anchor_objects_set:
                continue
            rotation_xyzw = get_rotation_xyzw(obj)
            poses = []
            for env_idx in range(num_envs):
                pos = layouts[env_idx].positions.get(obj)
                if pos is None:
                    break
                poses.append(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            else:
                obj.set_initial_pose(PosePerEnv(poses=poses))

    def _modify_recorder_cfg_dataset_filename(self, recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
        """Modify the recorder dataset filename to include the timestamp and rank."""
        base = getattr(recorder_cfg, "dataset_filename", "dataset")
        recorder_cfg.dataset_filename = (
            f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{get_local_rank()}"
        )
        return recorder_cfg

    # This method gives the arena environment a chance to modify the environment configuration.
    # This is a workaround to allow user to gradually move to the new configuration system.
    # THE ORDER MATTERS HERE.
    # THIS WILL BE REMOVED IN THE FUTURE.
    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        """Modify the environment configuration."""
        if self.arena_env.task is not None:
            env_cfg = self.arena_env.task.modify_env_cfg(env_cfg)
        if self.arena_env.embodiment is not None:
            env_cfg = self.arena_env.embodiment.modify_env_cfg(env_cfg)
        env_cfg = self.arena_env.scene.modify_env_cfg(env_cfg)
        return env_cfg

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
                metrics=metrics,
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
                # metrics=metrics,
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
            from isaaclab_arena.environments.isaaclab_arena_manager_based_env import ArenaPhysicsCfg

            env_cfg.sim.physics = getattr(ArenaPhysicsCfg(), presets)

            # Set replicate_physics for shared physics representations.
            # For Newton, wihotut this flag, the simulation initialization
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
