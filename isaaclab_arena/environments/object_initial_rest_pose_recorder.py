# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record each scene object's first resting position in an episode."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.physics_settle import (
    DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
    DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    compute_objects_settled_mask,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from isaaclab_arena.environments.arena_world import ArenaWorld


@configclass
class ObjectInitialRestPoseRecorderCfg:
    """Configure when the environment captures an object's initial resting position."""

    consecutive_steps: int = 5
    """Consecutive control steps below both speed thresholds before recording."""

    linear_velocity_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD
    """Maximum linear speed in m/s; deformables use the 90th percentile of nodal speeds."""

    angular_velocity_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD
    """Maximum angular speed in rad/s for rigid objects."""


class ObjectInitialRestPoseRecorder:
    """Own episode reference positions for the scene's rigid and deformable objects.

    The environment updates this recorder after each control step and resets selected
    environments at episode boundaries. Reading a reference does not advance recording.
    """

    def __init__(self, scene: InteractiveScene, arena_world: ArenaWorld, cfg: ObjectInitialRestPoseRecorderCfg):
        assert (
            isinstance(cfg.consecutive_steps, int)
            and not isinstance(cfg.consecutive_steps, bool)
            and cfg.consecutive_steps > 0
        ), "consecutive_steps must be a positive integer."
        for threshold in (cfg.linear_velocity_threshold, cfg.angular_velocity_threshold):
            assert math.isfinite(threshold) and threshold > 0, "Velocity thresholds must be finite and positive."
        self._scene = scene
        self._arena_world = arena_world
        self._cfg = cfg
        self._positions: dict[str, torch.Tensor] = {}
        self._recorded: dict[str, torch.Tensor] = {}
        self._consecutive_rest_steps: dict[str, torch.Tensor] = {}
        for object_name in (*scene.rigid_objects, *scene.deformable_objects):
            self._positions[object_name] = torch.full((scene.num_envs, 3), float("nan"), device=scene.device)
            self._recorded[object_name] = torch.zeros(scene.num_envs, dtype=torch.bool, device=scene.device)
            self._consecutive_rest_steps[object_name] = torch.zeros(
                scene.num_envs, dtype=torch.long, device=scene.device
            )

    def update(self) -> None:
        """Capture each object's first position after the configured consecutive resting steps."""
        for object_name, recorded in self._recorded.items():
            if bool(recorded.all()):
                continue
            at_rest = compute_objects_settled_mask(
                self._arena_world,
                self._scene,
                [object_name],
                self._cfg.linear_velocity_threshold,
                self._cfg.angular_velocity_threshold,
            )
            rest_steps = self._consecutive_rest_steps[object_name]
            rest_steps.copy_(torch.where(at_rest, (rest_steps + 1).clamp(max=self._cfg.consecutive_steps), 0))
            newly_recorded = ~recorded & (rest_steps >= self._cfg.consecutive_steps)
            if bool(newly_recorded.any()):
                positions = self._arena_world.get_position_w(object_name)
                self._positions[object_name][newly_recorded] = positions[newly_recorded]
                recorded |= newly_recorded

    def get(self, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return snapshots of initial world positions and the mask of environments with a reference.

        Args:
            object_name: Scene key of a rigid or deformable object.

        Returns:
            Positions shaped (num_envs, 3), with NaN for missing references, and a Boolean mask.
        """
        return self._positions[object_name].clone(), self._recorded[object_name].clone()

    def reset(self, env_ids=None) -> None:
        """Clear reference positions and resting streaks for selected environments, or all when omitted."""
        selected_env_ids = slice(None) if env_ids is None else env_ids
        for object_name in self._positions:
            self._positions[object_name][selected_env_ids] = float("nan")
            self._recorded[object_name][selected_env_ids] = False
            self._consecutive_rest_steps[object_name][selected_env_ids] = 0
