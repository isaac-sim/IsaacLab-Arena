# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from dataclasses import MISSING
from typing import Literal

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.events import randomize_poses_and_align_auxiliary_assets
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import objects_in_proximity
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class AssemblyTask(TaskBase):
    """
    Assembly task where an object needs to be assembled with a base object, like peg insert, gear mesh, etc.
    """

    def __init__(
        self,
        fixed_asset: Asset,
        held_asset: Asset,
        auxiliary_asset_list: list[Asset],
        background_scene: Asset,
        episode_length_s: float | None = None,
        max_x_separation: float = 0.020,
        max_y_separation: float = 0.020,
        max_z_separation: float = 0.020,
        task_description: str | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        min_separation: float = 0.10,
        randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.fixed_asset = fixed_asset
        self.held_asset = held_asset
        self.auxiliary_asset_list = auxiliary_asset_list
        self.background_scene = background_scene
        self.scene_config = None
        # We use specialize randomization at reset for this task. So disable default pose resets.
        self.disable_default_pose_resets()
        self.events_cfg = EventsCfg(
            pose_range=pose_range if pose_range is not None else {},
            min_separation=min_separation,
            asset_cfgs=[SceneEntityCfg(asset.name) for asset in [self.fixed_asset, self.held_asset]],
            fixed_asset_cfg=SceneEntityCfg(self.fixed_asset.name),
            auxiliary_asset_cfgs=[SceneEntityCfg(asset.name) for asset in self.auxiliary_asset_list],
            randomization_mode=randomization_mode,
        )
        self.termination_cfg = self._make_termination_cfg(
            max_x_separation=max_x_separation,
            max_y_separation=max_y_separation,
            max_z_separation=max_z_separation,
        )
        self.task_description = (
            f"Assemble the {self.held_asset.name} with the {self.fixed_asset.name}"
            if task_description is None
            else task_description
        )

    def disable_default_pose_resets(self):
        for asset in [self.fixed_asset, self.held_asset, *self.auxiliary_asset_list]:
            print(f"Disabling reset pose for asset: {asset.name}")
            asset.disable_reset_pose()

    def get_scene_cfg(self):
        """Get scene configuration."""
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def _make_termination_cfg(
        self,
        max_x_separation: float,
        max_y_separation: float,
        max_z_separation: float,
    ):
        """
        Create termination configuration for the assembly task.

        Args:
            max_x_separation: Maximum allowed separation in x-axis for success.
            max_y_separation: Maximum allowed separation in y-axis for success.
            max_z_separation: Maximum allowed separation in z-axis for success.

        Returns:
            TerminationsCfg: The termination configuration.
        """
        success = TerminationTermCfg(
            func=objects_in_proximity,
            params={
                "object_cfg": SceneEntityCfg(self.held_asset.name),
                "target_object_cfg": SceneEntityCfg(self.fixed_asset.name),
                "max_x_separation": max_x_separation,  # Tolerance for assembly alignment
                "max_y_separation": max_y_separation,
                "max_z_separation": max_z_separation,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_asset.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        """Get events configuration for assembly task."""
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        return FactoryAssemblyMimicEnvCfg()

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            ObjectMovedRateMetric(self.held_asset),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Get viewer configuration to look at the held asset.

        Camera is positioned at right-back-top of the object for better view of assembly operations.
        """
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_asset,
            offset=np.array([1.5, -0.5, 1.0]),  # Rotated 180° around z-axis from original view
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """
    Configuration for assembly task events.
    """

    reset_all: EventTermCfg = MISSING
    randomize_asset_positions: EventTermCfg = MISSING

    def __init__(
        self,
        pose_range: dict[str, tuple[float, float]],
        min_separation: float,
        asset_cfgs: list[SceneEntityCfg],
        fixed_asset_cfg: SceneEntityCfg,
        auxiliary_asset_cfgs: list[SceneEntityCfg],
        randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
    ):
        self.reset_all = EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
        )

        self.randomize_asset_positions = EventTermCfg(
            func=randomize_poses_and_align_auxiliary_assets,
            mode="reset",
            params={
                "pose_range": pose_range,
                "min_separation": min_separation,
                "asset_cfgs": asset_cfgs,
                "fixed_asset_cfg": fixed_asset_cfg,
                "auxiliary_asset_cfgs": auxiliary_asset_cfgs,
                "randomization_mode": randomization_mode,
            },
        )


class FactoryAssemblyMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for assembly task.

    Note:
        This is a base configuration class. Specific assembly tasks
        (e.g., PegInsert, GearMesh) should create their own subclasses with
        appropriate asset names.
    """

    embodiment_name: str = MISSING
    fixed_asset_name: str = MISSING
    held_asset_name: str = MISSING
    assist_asset_list_names: list[str] = MISSING
