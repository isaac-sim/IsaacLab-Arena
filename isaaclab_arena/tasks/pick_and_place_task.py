# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections.abc import Callable
from dataclasses import MISSING
from functools import partial
from typing import Literal

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.predicates.object_settling import objects_settled
from isaaclab_arena.tasks.predicates.spatial import (
    object_is_above_height,
    object_is_below_height,
    object_on_destination,
    object_on_destination_by_geometry,
    objects_in_proximity,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition
from isaaclab_arena.tasks.terminations import SuccessMode, check_success
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass

PickAndPlaceSuccessStrategy = Literal["auto", "contact", "geometry", "proximity"]


@agent_ready
@register_task
class PickAndPlaceTask(TaskBase):
    """Pick-and-place task. Success fires when the pick-up object contacts the destination
    with low velocity and, when ``max_separation`` is set, is within axis-aligned proximity
    of the destination. Objects without contact-sensor support use geometry success instead.
    Failure (object_dropped) fires when the object falls below the background's ``object_min_z``.

    The default Mimic cfg is ``PickPlaceMimicEnvCfg``. When a task needs a different cfg
    shape (different arm subtask sequences, different per-subtask numerical knobs,
    bespoke fields), pass ``mimic_env_cfg_factory`` to inject a custom ``MimicEnvCfg``::

        def _factory(arm_mode):
            return MyCustomMimicEnvCfg(arm_mode=arm_mode, ...)

        PickAndPlaceTask(..., mimic_env_cfg_factory=_factory)

    The factory receives ``arm_mode`` from the env builder and returns a constructed cfg.
    """

    def __init__(
        self,
        pick_up_object: ObjectBase,
        destination_location: Asset,
        background_scene: Asset,
        destination_object: Asset | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        force_threshold: float = 0.1,
        velocity_threshold: float = 0.1,
        max_separation: tuple[float, float, float] | None = None,
        success_strategy: PickAndPlaceSuccessStrategy = "auto",
        support_tolerance: float = 0.03,
        containment_margin: float = 0.01,
        containment_fraction_threshold: float = 0.5,
        mimic_env_cfg_factory: Callable[[ArmMode], MimicEnvCfg] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object = pick_up_object
        self.destination_object = destination_object
        self.background_scene = background_scene
        self.destination_location = destination_location
        self.force_threshold = force_threshold
        self.velocity_threshold = velocity_threshold
        if max_separation is not None:
            assert len(max_separation) == 3, f"max_separation must be (x, y, z), got {max_separation!r}"
        self.max_separation = max_separation
        self.support_tolerance = support_tolerance
        self.containment_margin = containment_margin
        self.containment_fraction_threshold = containment_fraction_threshold
        self.success_strategy = self._resolve_success_strategy(success_strategy)
        self.contact_sensor_name = (
            f"contact_sensor_{pick_up_object.name}" if self.success_strategy == "contact" else None
        )
        self.scene_config = self.make_scene_cfg()
        self.mimic_env_cfg_factory = mimic_env_cfg_factory
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Pick up the {pick_up_object.name}, and place it into the {destination_location.name}"
            if task_description is None
            else task_description
        )

    def _resolve_success_strategy(self, success_strategy: PickAndPlaceSuccessStrategy) -> str:
        """Resolve ``auto`` success based on object capabilities."""
        if success_strategy not in ("auto", "contact", "geometry", "proximity"):
            raise ValueError(
                "PickAndPlaceTask success_strategy must be one of 'auto', 'contact', 'geometry', or 'proximity'; "
                f"got {success_strategy!r}."
            )
        supports_contact = self.pick_up_object.supports_contact_sensor()
        if success_strategy == "auto":
            success_strategy = "contact" if supports_contact else "geometry"
        if success_strategy == "contact" and not supports_contact:
            raise ValueError(f"Object '{self.pick_up_object.name}' does not support contact-sensor success.")
        if success_strategy == "proximity" and self.max_separation is None:
            self.max_separation = (0.08, 0.08, 0.08)
        return success_strategy

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the object it picks up and the location it places onto."""
        self._apply_reachability_constraints([self.pick_up_object, self.destination_location])

    def make_scene_cfg(self):
        if self.success_strategy != "contact":
            return None
        assert self.contact_sensor_name is not None
        contact_sensor_cfg = self.pick_up_object.get_contact_sensor_cfg(
            contact_against_object=self.destination_location,
        )
        scene_cfg_type = make_configclass(
            "SceneCfg",
            [(self.contact_sensor_name, type(contact_sensor_cfg), contact_sensor_cfg)],
        )
        return scene_cfg_type()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        predicates = []
        if self.success_strategy == "contact":
            assert self.contact_sensor_name is not None
            predicates.append(
                TerminationTermCfg(
                    func=object_on_destination,
                    params={
                        "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                        "contact_sensor_cfg": SceneEntityCfg(self.contact_sensor_name),
                        "force_threshold": self.force_threshold,
                        "velocity_threshold": self.velocity_threshold,
                    },
                ),
            )
        elif self.success_strategy == "geometry":
            predicates.append(
                TerminationTermCfg(
                    func=object_on_destination_by_geometry,
                    params={
                        "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                        "target_object_cfg": SceneEntityCfg(self.destination_location.name),
                        "velocity_threshold": self.velocity_threshold,
                        "support_tolerance": self.support_tolerance,
                        "containment_margin": self.containment_margin,
                        "containment_fraction_threshold": self.containment_fraction_threshold,
                    },
                )
            )
        if self.max_separation is not None and self.success_strategy != "geometry":
            # TODO(qianl): replace objects_in_proximity with object_centroid_in_proximity
            # for tighter container placement checks.
            # TODO (qianl): current implementation doesn't support ObjectReference as target_object_cfg.
            max_x_separation, max_y_separation, max_z_separation = self.max_separation
            proximity_params = {
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "target_object_cfg": SceneEntityCfg(self.destination_location.name),
                "max_x_separation": max_x_separation,
                "max_y_separation": max_y_separation,
                "max_z_separation": max_z_separation,
            }
            if self.success_strategy == "proximity":
                proximity_params["velocity_threshold"] = self.velocity_threshold
            predicates.append(
                TerminationTermCfg(
                    func=objects_in_proximity,
                    params=proximity_params,
                )
            )
        success = TerminationTermCfg(
            func=check_success,
            params={
                "mode": SuccessMode.ALL,
                "predicates": predicates,
            },
        )
        object_dropped = TerminationTermCfg(
            func=object_is_below_height,
            params={
                "object_name": self.pick_up_object.name,
                "minimum_height": self.background_scene.object_min_z,
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Build the Mimic env cfg for this task.

        If ``mimic_env_cfg_factory`` was passed at construction, invoke it with
        ``arm_mode`` and return its result. Otherwise build the default
        ``PickPlaceMimicEnvCfg``.
        """
        if self.success_strategy != "contact":
            raise NotImplementedError("Mimic data generation requires contact-sensor pick-and-place success.")
        if self.mimic_env_cfg_factory is not None:
            return self.mimic_env_cfg_factory(arm_mode)
        destination_location_name = (
            self.destination_object.name if self.destination_object is not None else self.destination_location.name
        )
        return PickPlaceMimicEnvCfg(
            arm_mode=arm_mode,
            pick_up_object_name=self.pick_up_object.name,
            destination_location_name=destination_location_name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

    def get_progress_objectives(self) -> list[ProgressObjective]:
        if self.success_strategy == "geometry":
            return [
                ProgressObjective(
                    name="pick_and_place",
                    predicate_groups=[
                        partial(
                            objects_settled,
                            object_names=[self.pick_up_object.name],
                        ),
                        partial(
                            object_is_above_height,
                            object_name=self.pick_up_object.name,
                            use_settled_state=True,
                        ),
                        partial(
                            object_on_destination_by_geometry,
                            object_cfg=SceneEntityCfg(self.pick_up_object.name),
                            target_object_cfg=SceneEntityCfg(self.destination_location.name),
                            velocity_threshold=self.velocity_threshold,
                            support_tolerance=self.support_tolerance,
                            containment_margin=self.containment_margin,
                            containment_fraction_threshold=self.containment_fraction_threshold,
                        ),
                    ],
                ),
            ]
        if self.success_strategy == "proximity":
            assert self.max_separation is not None
            max_x_separation, max_y_separation, max_z_separation = self.max_separation
            return [
                ProgressObjective(
                    name="pick_and_place",
                    predicate_groups=[
                        partial(
                            objects_settled,
                            object_names=[self.pick_up_object.name],
                        ),
                        partial(
                            object_is_above_height,
                            object_name=self.pick_up_object.name,
                            use_settled_state=True,
                        ),
                        partial(
                            objects_in_proximity,
                            object_cfg=SceneEntityCfg(self.pick_up_object.name),
                            target_object_cfg=SceneEntityCfg(self.destination_location.name),
                            max_x_separation=max_x_separation,
                            max_y_separation=max_y_separation,
                            max_z_separation=max_z_separation,
                            velocity_threshold=self.velocity_threshold,
                        ),
                    ],
                ),
            ]
        assert self.contact_sensor_name is not None
        return [
            ProgressObjective(
                name="pick_and_place",
                predicate_groups=[
                    partial(
                        objects_settled,
                        object_names=[self.pick_up_object.name],
                    ),
                    partial(
                        object_is_above_height,
                        object_name=self.pick_up_object.name,
                        use_settled_state=True,
                    ),
                    partial(
                        object_on_destination,
                        object_cfg=SceneEntityCfg(self.pick_up_object.name),
                        contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
                        force_threshold=self.force_threshold,
                        velocity_threshold=self.velocity_threshold,
                    ),
                ],
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )

    @classmethod
    def success_state_transition(cls, pick_up_object: str, destination_location: str, **_) -> TaskTransition:
        """Success means the picked object ends up in an ``on`` relation with the destination."""
        # Note: with the current AABB-based object solver, placing an object ``on`` an open container
        # and letting it fall is equivalent to it being ``in`` the container, so a single ``on``
        # relation covers both surfaces and containers.
        relation = ObjectRelationLibraryRegistry().get_object_relation_by_name("on")
        return TaskTransition(
            subject=pick_up_object,
            effects=(Relocate(subject=pick_up_object, relation=relation.name, target=destination_location),),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class PickPlaceMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Pick and Place env.
    """

    arm_mode: ArmMode = ArmMode.SINGLE_ARM

    pick_up_object_name: str = "pick_up_object"

    destination_location_name: str = "destination_location"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_pickplace_isaac_lab_task_D0"
        # Use default mimic datagen config parameters
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                # TODO(alexmillane, 2025.09.02): This is currently broken. FIX.
                # We need a way to pass in a reference to an object that exists in the
                # scene.
                object_ref=self.destination_location_name,
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        if self.arm_mode == ArmMode.SINGLE_ARM:
            self.subtask_configs["robot"] = subtask_configs
        # We need to add the left and right subtasks for GR1.
        elif self.arm_mode in [ArmMode.LEFT, ArmMode.RIGHT]:
            self.subtask_configs[self.arm_mode.value] = subtask_configs
            # EEF on opposite side (arm is static)
            subtask_configs = []
            subtask_configs.append(
                SubTaskConfig(
                    # Each subtask involves manipulation with respect to a single object frame.
                    object_ref=self.pick_up_object_name,
                    # Corresponding key for the binary indicator in "datagen_info" for completion
                    subtask_term_signal=None,
                    # Time offsets for data generation when splitting a trajectory
                    subtask_term_offset_range=(0, 0),
                    # Selection strategy for source subtask segment
                    selection_strategy="nearest_neighbor_object",
                    # Optional parameters for the selection strategy function
                    selection_strategy_kwargs={"nn_k": 3},
                    # Amount of action noise to apply during this subtask
                    action_noise=0.005,
                    # Number of interpolation steps to bridge to this subtask segment
                    num_interpolation_steps=0,
                    # Additional fixed steps for the robot to reach the necessary pose
                    num_fixed_steps=0,
                    # If True, apply action noise during the interpolation phase and execution
                    apply_noise_during_interpolation=False,
                )
            )
            self.subtask_configs[self.arm_mode.get_other_arm().value] = subtask_configs

        else:
            raise ValueError(f"Embodiment arm mode {self.arm_mode} not supported")


@configclass
class G1PickAndPlaceMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for the G1 humanoid pick-and-place tasks.

    Encodes G1-specific cfg shape: 3 subtasks per arm
    (idle / grasp_and_idle / final), a 4-phase nav body subtask sequence
    (navigate_to_table / navigate_turn_inplace / navigate_to_bin / final), and
    G1-specific per-subtask numerical tunings. Used by the locomanip env via
    ``PickAndPlaceTask(mimic_env_cfg_factory=...)``.
    """

    pick_up_object_name: str = MISSING
    destination_location_name: str = MISSING
    arm_mode: ArmMode = ArmMode.DUAL_ARM

    def __post_init__(self):
        from isaaclab_arena_g1.g1_env.mdp.recorders.g1_locomanip_recorder_cfg import G1LocomanipRecorderManagerCfg

        # post init of parents
        super().__post_init__()

        # Hardcoded right + left + body subtask shapes assume both arms drive the demo;
        # other arm modes would silently produce a cfg with the wrong subtask shape.
        if self.arm_mode != ArmMode.DUAL_ARM:
            raise ValueError(f"G1PickAndPlaceMimicEnvCfg only supports ArmMode.DUAL_ARM; got {self.arm_mode}")

        self.datagen_config.name = (
            f"locomanip_pick_and_place_{self.pick_up_object_name}_to_{self.destination_location_name}_D0"
        )
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1
        self.datagen_config.use_navigation_controller = True

        self.mimic_recorder_config = G1LocomanipRecorderManagerCfg()

        # Right arm subtasks
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="idle_right",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_and_idle_right",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs

        # Left arm subtasks
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="idle_left",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_and_idle_left",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs

        # Body subtasks (used for navigation)
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="navigate_to_table",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="navigate_turn_inplace",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="navigate_to_bin",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["body"] = subtask_configs
