# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Maple-table pick-and-place, instrumented with raw per-step state recording.

Identical to ``pick_and_place_maple_table`` in every respect that affects the policy — same scene,
same embodiment, same task, same terminations — so results transfer to the published benchmark. The
only addition is a set of raw per-step recorder terms, from which failure margins are derived
offline.

Registered as ``etl_pnp_maple_table``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
    PickAndPlaceMapleTableEnvironment,
    PickAndPlaceMapleTableEnvironmentCfg,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class EtlPnpMapleTableEnvironmentCfg(PickAndPlaceMapleTableEnvironmentCfg):
    """Configure raw-state recording and released-placement evaluation."""

    sim_state_stride: int = 15
    """Steps between restorable-state captures. 15 = 1 s at the 15 Hz control rate; failures localise
    to tens of steps, so finer capture buys nothing and costs a lot of storage."""

    success_predicate_version: str = "etl_released_placement_v1"
    placement_max_horizontal_offset_m: float = 0.02
    placement_vertical_offset_range_m: tuple[float, float] = (0.06, 0.115)
    placement_max_axis_tilt_rad: float = 0.5
    placement_max_linear_speed_m_s: float = 0.03
    placement_max_angular_speed_rad_s: float = 0.2
    placement_max_open_joint_position_rad: float = 0.1
    placement_min_end_effector_distance_m: float = 0.15
    placement_min_contact_force_n: float = 0.1
    placement_dwell_steps: int = 8
    released_contact_predicate_version: str = "etl_released_contact_v1"
    released_contact_max_horizontal_offset_m: float = 0.12
    released_contact_max_linear_speed_m_s: float = 0.1
    released_contact_min_contact_force_n: float = 1.0
    pick_up_object_mass_kg: float | None = None
    """Optional measured mass override. ``None`` preserves the source USD exactly."""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.sim_state_stride > 0, "sim_state_stride must be positive."
        assert self.success_predicate_version, "success_predicate_version must not be empty."
        assert self.placement_max_horizontal_offset_m > 0.0
        min_vertical_offset, max_vertical_offset = self.placement_vertical_offset_range_m
        assert 0.0 <= min_vertical_offset < max_vertical_offset
        assert 0.0 <= self.placement_max_axis_tilt_rad <= math.pi / 2
        assert self.placement_max_linear_speed_m_s > 0.0
        assert self.placement_max_angular_speed_rad_s > 0.0
        assert self.placement_max_open_joint_position_rad >= 0.0
        assert self.placement_min_end_effector_distance_m > 0.0
        assert self.placement_min_contact_force_n > 0.0
        assert self.placement_dwell_steps > 0
        assert self.released_contact_predicate_version
        assert self.released_contact_max_horizontal_offset_m > 0.0
        assert self.released_contact_max_linear_speed_m_s > 0.0
        assert self.released_contact_min_contact_force_n > 0.0
        assert self.pick_up_object_mass_kg is None or self.pick_up_object_mass_kg > 0.0


@register_environment
class EtlPnpMapleTableEnvironment(ArenaEnvironmentFactory[EtlPnpMapleTableEnvironmentCfg]):
    """Base environment plus raw-state recording.

    Composes the base factory rather than subclassing it: registration reads the config type from a
    direct ``ArenaEnvironmentFactory[Cfg]`` base, so a factory that inherits another factory has two
    generic bases and fails to register.
    """

    name: str = "etl_pnp_maple_table"
    _legacy_argparse_cfg_type = EtlPnpMapleTableEnvironmentCfg

    def build(self, cfg: EtlPnpMapleTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

        from isaaclab_arena.recording.raw_state_terms import make_raw_state_recorder_cfg
        from isaaclab_arena.tasks.predicates.released_placement import ReleasedPlacementDwell
        from isaaclab_arena.variations.object_yaw_variation import ObjectYawVariation

        arena_env = PickAndPlaceMapleTableEnvironment().build(cfg)
        if cfg.pick_up_object_mass_kg is not None:
            arena_env.scene.assets[cfg.pick_up_object].object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(
                mass=cfg.pick_up_object_mass_kg
            )

        # Scene entity names are the asset-registry names, so cfg fields address the scene directly.
        # Raw channels say what happened; sim_state says where we were, in the form reset_to accepts.
        # Recovery generation needs the second.
        recorder_cfg = make_raw_state_recorder_cfg(
            object_name=cfg.pick_up_object,
            destination_name=cfg.destination_location,
            sim_state_stride=cfg.sim_state_stride,
        )
        arena_env.task.get_recorder_term_cfg = lambda: recorder_cfg  # type: ignore[method-assign]

        min_vertical_offset, max_vertical_offset = cfg.placement_vertical_offset_range_m
        arena_env.task.termination_cfg.success = TerminationTermCfg(
            func=ReleasedPlacementDwell,
            params={
                "object_cfg": SceneEntityCfg(cfg.pick_up_object),
                "destination_cfg": SceneEntityCfg(cfg.destination_location),
                "contact_sensor_cfg": SceneEntityCfg(arena_env.task.contact_sensor_name),
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["finger_joint"],
                    body_names=["base_link"],
                    preserve_order=True,
                ),
                "max_horizontal_offset": cfg.placement_max_horizontal_offset_m,
                "min_vertical_offset": min_vertical_offset,
                "max_vertical_offset": max_vertical_offset,
                "max_axis_tilt": cfg.placement_max_axis_tilt_rad,
                "max_linear_speed": cfg.placement_max_linear_speed_m_s,
                "max_angular_speed": cfg.placement_max_angular_speed_rad_s,
                "max_open_joint_position": cfg.placement_max_open_joint_position_rad,
                "min_end_effector_distance": cfg.placement_min_end_effector_distance_m,
                "min_contact_force": cfg.placement_min_contact_force_n,
                "dwell_steps": cfg.placement_dwell_steps,
            },
        )
        arena_env.task.success_predicate_version = cfg.success_predicate_version

        # Offer object yaw as a recorded factor. Disabled by default like every other variation, so
        # the environment matches the published benchmark until explicitly turned on with
        # `<object>.object_yaw_<object>.enabled=true`. Attached after build but before the builder
        # composes the events cfg, which is when variations are collected.
        arena_env.scene.assets[cfg.pick_up_object].add_variation(ObjectYawVariation(cfg.pick_up_object))
        return arena_env
