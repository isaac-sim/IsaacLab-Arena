# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena task wrapper for Isaac Cap cable-routing semantics."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.cable import Cable
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from .geometry import cable_route_success_from_geometry


def cable_route_success(
    env,
    *,
    cable_asset_name: str,
    peg_asset_names: tuple[str, ...],
    route_peg_indices: tuple[int, ...],
    route_directions: tuple[float, ...],
) -> torch.Tensor:
    """Return cable-route completion from native scene-asset state."""
    cable_points_w = env.scene[cable_asset_name].data.segment_pose_w.torch[..., :3]
    peg_positions_w = torch.stack(
        [env.scene[peg_name].data.root_pos_w.torch for peg_name in peg_asset_names],
        dim=1,
    )
    return cable_route_success_from_geometry(
        cable_points_w,
        peg_positions_w,
        route_peg_indices=route_peg_indices,
        route_directions=route_directions,
    )


@configclass
class CableRoutingEventsCfg:
    """Reset every native scene entity to its authored default."""

    reset_scene: EventTerm = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class CableRoutingTerminationsCfg:
    """Cable-route success and timeout terms."""

    success: DoneTerm = MISSING
    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)


class CableRoutingTask(TaskBase):
    """Route one cable around the configured sequence of pegs."""

    def __init__(
        self,
        *,
        cable: Cable,
        pegs: Sequence[ObjectBase],
        route_peg_indices: tuple[int, ...],
        route_directions: tuple[float, ...],
        task_description: str,
        episode_length_s: float = 3600.0,
        viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Configure cable routing and its geometric success check.

        Args:
            cable: Native Arena cable asset to evaluate.
            pegs: Every peg present in the scene.
            route_peg_indices: Indices of the pegs in the required route.
            route_directions: Winding direction per route peg; zero accepts either.
            task_description: Language instruction exposed by Arena.
            episode_length_s: Episode timeout in seconds.
            viewer_lookat: Environment-local viewer target.
        """
        assert isinstance(cable, Cable), "cable must be an Arena Cable asset."
        assert pegs, "At least one peg is required."
        assert len(route_peg_indices) == len(route_directions), "Each route peg must have one direction."
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self._events_cfg = CableRoutingEventsCfg()
        self._terminations_cfg = CableRoutingTerminationsCfg(
            success=DoneTerm(
                func=cable_route_success,
                params={
                    "cable_asset_name": cable.name,
                    "peg_asset_names": tuple(peg.name for peg in pegs),
                    "route_peg_indices": route_peg_indices,
                    "route_directions": route_directions,
                },
            )
        )
        self._viewer_lookat = viewer_lookat

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self._terminations_cfg

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.25, -1.10, 1.55), lookat=self._viewer_lookat)
