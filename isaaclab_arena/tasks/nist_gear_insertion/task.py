# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST gear insertion task for Arena policy evaluation."""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from . import geometry as gear_insertion_geometry
from .terminations import gear_mesh_insertion_success


@configclass
class GearInsertionGeometryCfg:
    """NIST board gear-insertion geometry shared by observations and success."""

    peg_base_offset: tuple[float, float, float] = (0.02025, 0.0, 0.0)
    peg_tip_offset: tuple[float, float, float] = (0.02025, 0.0, 0.025)
    held_gear_base_offset: tuple[float, float, float] = (0.02025, 0.0, 0.0)
    gear_peg_height: float = 0.02
    success_z_fraction: float = 0.30
    xy_threshold: float = 0.0025


@register_task
class NistGearInsertionTask(TaskBase):
    """Evaluation task for inserting the medium NIST gear onto the board peg."""

    def __init__(
        self,
        held_gear: Asset,
        background_scene: Asset,
        gear_base_asset: Asset,
        geometry_cfg: GearInsertionGeometryCfg | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.held_gear = held_gear
        self.background_scene = background_scene
        self.gear_base_asset = gear_base_asset
        self.geometry_cfg = geometry_cfg if geometry_cfg is not None else GearInsertionGeometryCfg()
        if self.task_description is None:
            self.task_description = "Insert the medium NIST gear onto the NIST gear base."

    def get_scene_cfg(self) -> Any:
        """Return no additional scene config."""

    def get_observation_cfg(self) -> GearInsertionObservationsCfg:
        """Return task-state observations for the gear and target peg."""
        geometry_cfg = self.geometry_cfg
        return GearInsertionObservationsCfg(
            gear_name=self.held_gear.name,
            board_name=self.gear_base_asset.name,
            peg_offset=geometry_cfg.peg_tip_offset,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
        )

    def get_rewards_cfg(self) -> Any:
        """Return no rewards for evaluation-only rollouts."""

    def get_termination_cfg(self) -> GearInsertionTerminationsCfg:
        """Return timeout, success, and object-drop terminations."""
        geometry_cfg = self.geometry_cfg
        success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_gear.name),
                "fixed_object_cfg": SceneEntityCfg(self.gear_base_asset.name),
                "gear_base_offset": geometry_cfg.peg_base_offset,
                "held_gear_base_offset": geometry_cfg.held_gear_base_offset,
                "gear_peg_height": geometry_cfg.gear_peg_height,
                "success_z_fraction": geometry_cfg.success_z_fraction,
                "xy_threshold": geometry_cfg.xy_threshold,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_gear.name),
            },
        )
        return GearInsertionTerminationsCfg(success=success, object_dropped=object_dropped)

    def get_events_cfg(self) -> Any:
        """Return no task-specific reset events."""

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        """Raise because this task currently only defines policy evaluation."""
        del arm_mode
        raise NotImplementedError("NIST gear insertion does not define a Mimic configuration yet.")

    def get_metrics(self) -> list[MetricBase]:
        """Return task metrics used during evaluation."""
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return a camera view focused on the held gear and peg area."""
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_gear,
            offset=np.array([1.5, -0.5, 1.0]),
        )


@configclass
class GearInsertionTerminationsCfg:
    """Termination terms for NIST gear insertion evaluation."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class GearInsertionTaskObsCfg(ObsGroup):
    """Task-state observations for the NIST gear and target peg."""

    gear_pos: ObsTerm = MISSING
    gear_quat: ObsTerm = MISSING
    fixed_base_pos: ObsTerm = MISSING
    fixed_base_quat: ObsTerm = MISSING
    peg_pos: ObsTerm = MISSING
    peg_delta: ObsTerm = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GearInsertionObservationsCfg:
    """Observation config for NIST gear insertion task state."""

    task_obs: GearInsertionTaskObsCfg = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: tuple[float, float, float],
        held_gear_base_offset: tuple[float, float, float],
    ):
        self.task_obs = GearInsertionTaskObsCfg()
        gear_cfg = SceneEntityCfg(gear_name)
        board_cfg = SceneEntityCfg(board_name)

        self.task_obs.gear_pos = ObsTerm(
            func=mdp_isaac_lab.root_pos_w,
            params={"asset_cfg": gear_cfg},
        )
        self.task_obs.gear_quat = ObsTerm(
            func=mdp_isaac_lab.root_quat_w,
            params={"make_quat_unique": True, "asset_cfg": gear_cfg},
        )
        self.task_obs.fixed_base_pos = ObsTerm(
            func=mdp_isaac_lab.root_pos_w,
            params={"asset_cfg": board_cfg},
        )
        self.task_obs.fixed_base_quat = ObsTerm(
            func=mdp_isaac_lab.root_quat_w,
            params={"make_quat_unique": True, "asset_cfg": board_cfg},
        )
        self.task_obs.peg_pos = ObsTerm(
            func=gear_insertion_geometry.peg_pos_in_env_frame,
            params={"board_cfg": board_cfg, "peg_offset": peg_offset},
        )
        self.task_obs.peg_delta = ObsTerm(
            func=gear_insertion_geometry.peg_delta_from_held_gear_base,
            params={
                "gear_cfg": gear_cfg,
                "board_cfg": board_cfg,
                "peg_offset": peg_offset,
                "held_gear_base_offset": held_gear_base_offset,
            },
        )
