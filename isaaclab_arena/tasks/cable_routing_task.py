# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cable-routing task and geometric success condition."""

from __future__ import annotations

import math
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
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

DEFAULT_ROUTE_RADIAL_CUTOFF = 0.05
DEFAULT_ROUTE_AXIAL_CUTOFF = 0.015
DEFAULT_ROUTE_COMPLETION_WINDING = 2.6
DEFAULT_ROUTE_MAXIMUM_COMPLETION_WINDING = 2.0 * math.pi + 0.25
DEFAULT_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH = 0.25


def cable_route_success_from_geometry(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    route_directions: Sequence[float],
    radial_cutoff: float = DEFAULT_ROUTE_RADIAL_CUTOFF,
    axial_cutoff: float = DEFAULT_ROUTE_AXIAL_CUTOFF,
    completion_winding: float = DEFAULT_ROUTE_COMPLETION_WINDING,
    maximum_completion_winding: float = DEFAULT_ROUTE_MAXIMUM_COMPLETION_WINDING,
    maximum_local_cable_length: float = DEFAULT_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH,
) -> torch.Tensor:
    """Evaluate an ordered sequence of directed cable windings around pegs.

    Args:
        cable_points_w: Ordered cable point positions with shape ``(N, S, 3)``.
        peg_positions_w: Ordered route peg positions with shape ``(N, P, 3)``.
        route_directions: Winding sign per peg; ``-1`` is counterclockwise and ``1`` is clockwise.
        radial_cutoff: Maximum radial distance for points local to a peg.
        axial_cutoff: Maximum absolute vertical distance for points local to a peg.
        completion_winding: Minimum directed winding angle in radians.
        maximum_completion_winding: Maximum directed winding angle in radians.
        maximum_local_cable_length: Maximum cable length assigned to one peg winding.

    Returns:
        Boolean success tensor with shape ``(N,)``.
    """
    assert (
        cable_points_w.ndim == 3 and cable_points_w.shape[-1] == 3
    ), f"Expected cable points with shape (N, S, 3), got {tuple(cable_points_w.shape)}."
    assert (
        peg_positions_w.ndim == 3 and peg_positions_w.shape[-1] == 3
    ), f"Expected peg positions with shape (N, P, 3), got {tuple(peg_positions_w.shape)}."
    assert (
        cable_points_w.shape[0] == peg_positions_w.shape[0]
    ), "Cable points and peg positions must contain the same number of environments."
    assert cable_points_w.shape[1] >= 2, "At least two ordered cable points are required."
    assert peg_positions_w.shape[1] == len(
        route_directions
    ), f"Expected {peg_positions_w.shape[1]} route directions, got {len(route_directions)}."
    assert len(route_directions) > 0, "At least one route peg is required."
    assert radial_cutoff > 0.0, "radial_cutoff must be positive."
    assert axial_cutoff > 0.0, "axial_cutoff must be positive."
    assert maximum_local_cable_length > 0.0, "maximum_local_cable_length must be positive."
    assert (
        maximum_completion_winding >= completion_winding
    ), "maximum_completion_winding must be greater than or equal to completion_winding."

    finite_geometry = torch.isfinite(cable_points_w).all(dim=(1, 2)) & torch.isfinite(peg_positions_w).all(dim=(1, 2))
    safe_cable_points = torch.where(finite_geometry[:, None, None], cable_points_w, torch.zeros_like(cable_points_w))
    safe_peg_positions = torch.where(
        finite_geometry[:, None, None],
        peg_positions_w,
        torch.zeros_like(peg_positions_w),
    )

    relative_xy = safe_cable_points[:, None, :, :2] - safe_peg_positions[:, :, None, :2]
    local_points = torch.linalg.vector_norm(relative_xy, dim=-1) <= radial_cutoff
    relative_z = safe_cable_points[:, None, :, 2] - safe_peg_positions[:, :, None, 2]
    local_points &= relative_z.abs() <= axial_cutoff

    angle = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
    angle_delta = angle[..., 1:] - angle[..., :-1]
    angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
    local_edges = local_points[..., :-1] & local_points[..., 1:]
    clockwise_winding = -torch.where(local_edges, angle_delta, 0.0).sum(dim=-1)

    previous_local_edges = torch.zeros_like(local_edges)
    previous_local_edges[..., 1:] = local_edges[..., :-1]
    local_span_count = (local_edges & ~previous_local_edges).sum(dim=-1)
    edge_lengths = torch.linalg.vector_norm(safe_cable_points[:, 1:] - safe_cable_points[:, :-1], dim=-1)
    local_cable_length = torch.where(local_edges, edge_lengths[:, None, :], 0.0).sum(dim=-1)
    route_geometry_valid = (local_span_count == 1) & (local_cable_length <= maximum_local_cable_length)

    route_directions_tensor = torch.as_tensor(
        route_directions,
        device=cable_points_w.device,
        dtype=cable_points_w.dtype,
    )
    directed_winding = clockwise_winding * route_directions_tensor[None, :]
    route_steps_complete = (
        route_geometry_valid
        & (directed_winding >= completion_winding)
        & (directed_winding <= maximum_completion_winding)
    )

    edge_indices = torch.arange(local_edges.shape[-1], device=cable_points_w.device)[None, None, :]
    first_local_edge = torch.where(local_edges, edge_indices, local_edges.shape[-1]).amin(dim=-1)
    last_local_edge = torch.where(local_edges, edge_indices, -1).amax(dim=-1)
    route_is_ordered = (last_local_edge[:, :-1] < first_local_edge[:, 1:]).all(dim=1)

    return finite_geometry & route_steps_complete.all(dim=1) & route_is_ordered


def cable_route_success(
    env,
    cable_asset_name: str,
    peg_asset_names: Sequence[str],
    route_directions: Sequence[float],
    radial_cutoff: float = DEFAULT_ROUTE_RADIAL_CUTOFF,
    axial_cutoff: float = DEFAULT_ROUTE_AXIAL_CUTOFF,
    completion_winding: float = DEFAULT_ROUTE_COMPLETION_WINDING,
    maximum_completion_winding: float = DEFAULT_ROUTE_MAXIMUM_COMPLETION_WINDING,
    maximum_local_cable_length: float = DEFAULT_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH,
) -> torch.Tensor:
    """Return cable-routing success for the configured scene entities."""
    cable_points_w = env.scene[cable_asset_name].data.segment_pose_w.torch[..., :3]
    peg_positions_w = torch.stack(
        [env.scene[peg_asset_name].data.root_pos_w.torch for peg_asset_name in peg_asset_names],
        dim=1,
    )
    return cable_route_success_from_geometry(
        cable_points_w=cable_points_w,
        peg_positions_w=peg_positions_w,
        route_directions=route_directions,
        radial_cutoff=radial_cutoff,
        axial_cutoff=axial_cutoff,
        completion_winding=completion_winding,
        maximum_completion_winding=maximum_completion_winding,
        maximum_local_cable_length=maximum_local_cable_length,
    )


@configclass
class CableRoutingEventsCfg:
    """Reset all scene entities and joint targets to their defaults."""

    reset_scene: EventTerm = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class CableRoutingTerminationsCfg:
    """Cable-route success and episode timeout terms."""

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = MISSING


@register_task
class CableRoutingTask(TaskBase):
    """Route one cable around an ordered sequence of pegs."""

    def __init__(
        self,
        cable: Cable,
        pegs: Sequence[ObjectBase],
        route_directions: Sequence[float],
        episode_length_s: float = 3600.0,
        radial_cutoff: float = DEFAULT_ROUTE_RADIAL_CUTOFF,
        axial_cutoff: float = DEFAULT_ROUTE_AXIAL_CUTOFF,
        completion_winding: float = DEFAULT_ROUTE_COMPLETION_WINDING,
        maximum_completion_winding: float = DEFAULT_ROUTE_MAXIMUM_COMPLETION_WINDING,
        maximum_local_cable_length: float = DEFAULT_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH,
        task_description: str | None = None,
        viewer_eye: tuple[float, float, float] = (1.25, -1.10, 1.55),
        viewer_lookat: tuple[float, float, float] | None = None,
    ) -> None:
        """Initialize an ordered cable-routing task.

        Args:
            cable: Cable scene asset evaluated for success.
            pegs: Route peg assets in required cable order.
            route_directions: Winding sign for each peg.
            episode_length_s: Episode timeout in seconds.
            radial_cutoff: Maximum radial distance for cable points local to a peg.
            axial_cutoff: Maximum vertical distance for cable points local to a peg.
            completion_winding: Minimum directed winding angle in radians.
            maximum_completion_winding: Maximum directed winding angle in radians.
            maximum_local_cable_length: Maximum cable length assigned to one peg.
            task_description: Optional language instruction.
            viewer_eye: Viewer camera position.
            viewer_lookat: Viewer camera target; defaults to the cable initial position.
        """
        assert isinstance(cable, Cable), "cable must be a Cable asset."
        assert len(pegs) > 0, "At least one route peg is required."
        assert all(isinstance(peg, ObjectBase) for peg in pegs), "Every route peg must be an ObjectBase asset."
        assert len(pegs) == len(route_directions), "Each route peg must have one winding direction."
        assert episode_length_s > 0.0, "episode_length_s must be positive."
        description = task_description or (
            "Route the cable counterclockwise around the first peg, then clockwise around the second peg, "
            "using both YAM manipulators."
        )
        super().__init__(episode_length_s=episode_length_s, task_description=description)
        self.cable = cable
        self.pegs = tuple(pegs)
        self.route_directions = tuple(float(direction) for direction in route_directions)
        self.viewer_eye = viewer_eye
        cable_initial_pose = cable.get_initial_pose()
        self.viewer_lookat = viewer_lookat or (
            cable_initial_pose.position_xyz if cable_initial_pose is not None else (0.0, 0.0, 0.0)
        )
        self._events_cfg = CableRoutingEventsCfg()
        self._terminations_cfg = CableRoutingTerminationsCfg(
            success=DoneTerm(
                func=cable_route_success,
                params={
                    "cable_asset_name": cable.name,
                    "peg_asset_names": tuple(peg.name for peg in self.pegs),
                    "route_directions": self.route_directions,
                    "radial_cutoff": radial_cutoff,
                    "axial_cutoff": axial_cutoff,
                    "completion_winding": completion_winding,
                    "maximum_completion_winding": maximum_completion_winding,
                    "maximum_local_cable_length": maximum_local_cable_length,
                },
            )
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self._terminations_cfg

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("CableRoutingTask does not provide a Mimic configuration.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=self.viewer_eye, lookat=self.viewer_lookat)
