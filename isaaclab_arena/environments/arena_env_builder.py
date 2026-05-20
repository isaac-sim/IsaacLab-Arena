# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import gymnasium as gym
from copy import deepcopy
from dataclasses import field, make_dataclass
from typing import Any

from isaaclab.devices.device_base import DeviceCfg, DevicesCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_teleop import IsaacTeleopCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase
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
from isaaclab_arena.variations.ledger import VariationLedger
from isaaclab_arena.variations.variation_base import VariationBase


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

    def _compose_variations_event_cfg(self):
        """Build a configclass instance holding an :class:`EventTermCfg` per enabled variation.

        Walks every variation on the scene (and, later, any env-level variation
        escape hatch), skips the disabled ones, and asks each enabled one for
        its event term via
        :meth:`~isaaclab_arena.variations.variation_base.VariationBase.build_event_cfg`.
        Returns ``None`` when nothing is enabled so
        :func:`combine_configclass_instances` skips it cleanly.

        The :class:`~isaaclab_arena.scene.scene.Scene` / asset surface now
        returns every variation regardless of state (see
        :meth:`Scene.get_variations`), so the ``enabled`` filter lives here —
        the same builder that consumes the inventory for the structured-Hydra
        layer (:meth:`get_variations_schema`).

        Raises:
            AssertionError: If two variations want the same event-term name
                (variations are responsible for uniquely namespacing their
                terms, typically by prefixing with the asset name).
        """
        variations = self.arena_env.scene.get_variations()
        fields: list[tuple[str, type, EventTermCfg]] = []
        seen: set[str] = set()
        for variation in variations:
            if not variation.enabled:
                continue
            event_name, event_cfg = variation.build_event_cfg(self.arena_env.scene)
            assert event_name not in seen, (
                f"Duplicate variation event term name '{event_name}'. "
                "Each variation must produce a unique name; consider prefixing with the asset name."
            )
            seen.add(event_name)
            fields.append((event_name, EventTermCfg, event_cfg))
        if not fields:
            return None
        VariationsEventCfg = make_configclass("VariationsEventCfg", fields)
        return VariationsEventCfg()

    def _iter_scene_variations(self) -> list[tuple[str, VariationBase]]:
        """Walk the scene and return ``(asset_name, variation)`` pairs for every variation.

        Used by both :meth:`_build_variations_schema` (where the asset name
        becomes the top-level Hydra field) and the forthcoming
        ``apply_hydra_variation_overrides`` (where it's the lookup key for
        writing composed values back). We resolve the asset name here rather
        than reading it off the variation because ``asset_name`` is an
        :class:`~isaaclab_arena.variations.object_color.ObjectColorVariation`
        implementation detail, not part of :class:`VariationBase`.
        """
        pairs: list[tuple[str, VariationBase]] = []
        for asset in self.arena_env.scene.assets.values():
            if not isinstance(asset, ObjectBase):
                continue
            for variation in asset.get_variations():
                pairs.append((asset.name, variation))
        return pairs

    def _populate_variation_ledger(self, ledger: VariationLedger) -> None:
        """Attach ``ledger`` to every *enabled* variation in the scene.

        Lives on the builder (rather than on :class:`VariationLedger`)
        so the ledger module does not have to import
        :class:`~isaaclab_arena.scene.scene.Scene` /
        :class:`~isaaclab_arena.assets.object_base.ObjectBase` — that
        back-edge cycles through the variation system, since
        ``ObjectBase.add_variation`` references ``VariationBase``. The
        builder is the natural owner of the walk anyway: it already
        owns :meth:`_iter_scene_variations` for the Hydra schema path
        and knows the same identity convention (``"{asset}.{variation}"``)
        that the override key paths use.

        Disabled variations are skipped because they never fire — they
        would just sit in the ledger as empty records and clutter the
        downstream sensitivity-analysis output.
        """
        for asset_name, variation in self._iter_scene_variations():
            if not variation.enabled:
                continue
            ledger.attach(f"{asset_name}.{variation.name}", variation)

    def _build_variations_schema(self, pairs: list[tuple[str, VariationBase]]) -> type:
        """Build a dynamic dataclass mirroring the scene's variations for Hydra.

        Each variation's existing ``*Cfg`` (e.g.
        :class:`~isaaclab_arena.variations.object_color.ObjectColorVariationCfg`)
        is used **as-is** as the per-variation schema node — it already carries
        the ``enabled`` field via :class:`VariationBaseCfg` and its nested
        sampler cfg (e.g.
        :class:`~isaaclab_arena.variations.sampler.UniformSamplerCfg`), so the
        Hydra override paths line up one-to-one with the cfg attribute paths
        (``<asset>.<variation>.enabled=true``,
        ``<asset>.<variation>.sampler.low=...``).

        The per-variation default-factory deep-copies the live ``variation.cfg``
        so each schema instance starts pre-populated from the variation's
        current state — useful for inspecting what knobs are available and for
        letting users override only what they want to change.

        Args:
            pairs: Output of :meth:`_iter_scene_variations`.

        Returns:
            A fresh ``VariationsCfg`` dataclass type with one field per asset,
            each holding a ``<AssetName>VariationsCfg`` dataclass whose fields
            are the per-variation cfgs.
        """
        per_asset: dict[str, list[tuple[str, type, Any]]] = {}
        for asset_name, variation in pairs:
            cfg_cls = type(variation.cfg)
            default_cfg = deepcopy(variation.cfg)
            per_asset.setdefault(asset_name, []).append(
                (variation.name, cfg_cls, field(default_factory=lambda d=default_cfg: deepcopy(d)))
            )

        asset_fields: list[tuple[str, type, Any]] = []
        for asset_name, variation_fields in per_asset.items():
            asset_cls = make_dataclass(self._asset_class_name(asset_name), variation_fields)
            asset_fields.append((asset_name, asset_cls, field(default_factory=asset_cls)))
        return make_dataclass("VariationsCfg", asset_fields)

    @staticmethod
    def _asset_class_name(asset_name: str) -> str:
        """``"cracker_box"`` -> ``"CrackerBoxVariationsCfg"``."""
        camel = "".join(part.capitalize() for part in asset_name.split("_"))
        return f"{camel}VariationsCfg"

    def get_variations_schema(self) -> type | None:
        """Return the dynamic :class:`dataclasses.dataclass` describing the scene's variations.

        Public entry point for the Hydra-driven variation layer. The class
        returned has one field per :class:`~isaaclab_arena.assets.object_base.ObjectBase`
        asset that owns at least one variation; each asset field's type is itself
        a dataclass whose fields are the variations attached to that asset, each
        typed as a dynamically-subclassed variation cfg with an extra
        ``enabled: bool`` field.

        Typical use::

            from omegaconf import OmegaConf
            schema_cls = env_builder.get_variations_schema()
            print(OmegaConf.to_yaml(OmegaConf.structured(schema_cls)))

        Returns ``None`` when the scene has no variations attached.
        """
        pairs = self._iter_scene_variations()
        if not pairs:
            return None
        return self._build_variations_schema(pairs)

    def compose_variations_cfg(self, hydra_overrides: list[str]) -> Any | None:
        """Compose Hydra override strings into a typed ``VariationsCfg`` instance.

        Step 1 of the two-step structured-config variation path. Builds the
        schema returned by :meth:`get_variations_schema`, registers it with
        Hydra's :class:`~hydra.core.config_store.ConfigStore`, composes the
        supplied overrides against it, and converts the result from the
        loosely-typed :class:`~omegaconf.DictConfig` form back into the
        dataclass tree the schema describes via
        :func:`omegaconf.OmegaConf.to_object`. The returned object is a
        ``VariationsCfg`` instance whose per-asset fields are themselves
        dataclass instances and whose leaf per-variation fields are typed
        ``*Cfg`` instances (e.g.
        :class:`~isaaclab_arena.variations.object_color.ObjectColorVariationCfg`),
        not :class:`~omegaconf.DictConfig`.

        Splitting this out from :meth:`apply_hydra_variation_overrides` lets
        the builder's *write-back* step deal with typed cfgs only — it never
        has to look up individual cfg fields by name, so adding a new
        variation (with a different cfg shape) doesn't touch the builder.

        Args:
            hydra_overrides: Hydra override strings, dotted-path syntax.
                See :meth:`apply_hydra_variation_overrides` for examples.

        Returns:
            The composed ``VariationsCfg`` instance, or ``None`` when the
            scene has no variations attached.

        Note:
            :class:`~hydra.core.global_hydra.GlobalHydra` is cleared on entry
            so this method is safe to call repeatedly in the same process
            (e.g. across cells of a notebook, or inside an eval-runner loop).
            See the open-questions section of ``2026_05_11_hydra_variation_plan.md``
            for the longer-running motivation behind this.
        """
        from hydra import compose, initialize
        from hydra.core.config_store import ConfigStore
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        pairs = self._iter_scene_variations()
        if not pairs:
            return None
        schema_cls = self._build_variations_schema(pairs)
        ConfigStore.instance().store(name="arena_variations_schema", node=schema_cls)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path=None):
            composed = compose(config_name="arena_variations_schema", overrides=hydra_overrides)
        return OmegaConf.to_object(composed)

    def apply_hydra_variation_overrides(self, hydra_overrides: list[str]) -> None:
        """Apply Hydra-style variation overrides to the scene's variations.

        Two-step:

        1. **Strings → typed cfg.** :meth:`compose_variations_cfg` composes
           the supplied override strings against the structured-config
           schema and returns a fully-typed ``VariationsCfg`` instance.
        2. **Typed cfg → live variation state.** For every
           ``(asset_name, variation)`` pair the builder knows about, the
           corresponding per-variation cfg is handed to
           :meth:`~isaaclab_arena.variations.variation_base.VariationBase.apply_cfg`,
           which replaces ``variation.cfg`` wholesale and rebuilds any
           derived live state (e.g. the live sampler).

        This split deliberately keeps the builder free of variation-specific
        field names: the variation cfg dataclass *is* the enumeration of
        the variation's tunable parameters, and
        :meth:`VariationBase.apply_cfg` is the abstraction boundary that
        knows how to map a cfg back onto its live variation. Adding a new
        variation (with its own ``*Cfg`` shape) requires no changes here.

        Args:
            hydra_overrides: Hydra override strings, dotted-path syntax
                mirroring the schema attribute paths (one level per asset,
                one per variation, then per cfg field). May be empty (no-op
                beyond a schema-defaults round-trip). Unknown asset /
                variation / field paths are rejected by Hydra's
                structured-config validator at compose time. Example::

                    env_builder.apply_hydra_variation_overrides([
                        "cracker_box.color.enabled=true",
                        "cracker_box.color.sampler.low=[0.2,0.2,0.0]",
                        "cracker_box.color.sampler.high=[1.0,1.0,0.0]",
                    ])
        """
        composed = self.compose_variations_cfg(hydra_overrides)
        if composed is None:
            return
        for asset_name, variation in self._iter_scene_variations():
            variation_cfg = getattr(getattr(composed, asset_name), variation.name)
            variation.apply_cfg(variation_cfg)

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
        variations_event_cfg = self._compose_variations_event_cfg()
        events_cfg = combine_configclass_instances(
            "EventsCfg",
            embodiment.get_events_cfg(),
            self.arena_env.scene.get_events_cfg(),
            task.get_events_cfg(),
            placement_event_cfg,
            variations_event_cfg,
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
                # NOTE(alexmillane, 2025-09-25): Metric + recorders excluded from mimic env,
                # I assume that they're not needed for the mimic env.
                # recorders=recorder_manager_cfg,
                # metrics=metrics,
                isaaclab_arena_env=isaaclab_arena_env,
                viewer=viewer_cfg,
            )

        # Variation recording layer. The ledger is created here (rather than
        # held on the builder) so it lives on the same cfg object as the rest
        # of the env state — callers can recover the input factors that drove
        # each draw via ``env.cfg.variation_ledger.records``. Listeners live on
        # :class:`~isaaclab_arena.variations.variation_base.VariationBase` and
        # survive subsequent sampler swaps, so attaching once after compose is
        # enough; this also runs before ``env_cfg_callback`` so callbacks can
        # observe / extend the ledger if they want to.
        env_cfg.variation_ledger = VariationLedger()
        self._populate_variation_ledger(env_cfg.variation_ledger)

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
