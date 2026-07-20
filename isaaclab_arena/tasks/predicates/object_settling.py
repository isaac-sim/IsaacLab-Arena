# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Object-settling predicate and recorder object.

The ``objects_settled`` predicate reports when all specified objects in the env come to a rest.
When an object settles, its initial resting position is recorded by the ``ObjectInitialRestPoseRecorder`` object.
Downstream predicates can read the positions via the ``get_object_initial_rest_state`` function.
Resetting and clearing of positions are handled by the progress tracker on env reset.
"""

from __future__ import annotations

import torch

from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_env,
    get_root_ang_vel_w,
    get_root_lin_vel_w,
    get_root_pos_w,
    select,
)


class ObjectInitialRestPoseRecorder:
    """Recorder object that works in conjunction with the ``objects_settled`` predicate to record the
    initial resting poses of scene objects and expose to downstream predicates.
    """

    def __init__(self, num_envs: int, device):
        self._num_envs = num_envs
        self._device = device
        self._entries: dict[str, dict[str, torch.Tensor]] = {}

    def _entry(self, name: str) -> dict[str, torch.Tensor]:
        """Get or create the ``{position, settled}`` record for one object."""

        entry = self._entries.get(name)
        if entry is None:
            entry = {
                "position": torch.full((self._num_envs, 3), float("nan"), device=self._device),
                "settled": torch.zeros(self._num_envs, dtype=torch.bool, device=self._device),
            }
            self._entries[name] = entry
        return entry

    def record(self, name: str, positions: torch.Tensor, settled: torch.Tensor) -> None:
        """Record object's world position for envs that just settled and weren't recorded yet."""

        entry = self._entry(name)
        new = settled & ~entry["settled"]
        if bool(new.any()):
            entry["position"] = torch.where(new.unsqueeze(-1), positions, entry["position"])
            entry["settled"] = entry["settled"] | new

    def get(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(position, settled)`` for object's recorded initial rest pose."""

        entry = self._entry(name)
        return entry["position"], entry["settled"]

    def reset(self, env_ids=None) -> None:
        """Clear recorded rest poses for ``env_ids`` (all envs if None)."""

        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        for entry in self._entries.values():
            entry["settled"][ids] = False
            entry["position"][ids] = float("nan")


def get_rest_pose_recorder(env) -> ObjectInitialRestPoseRecorder:
    """Return the ``ObjectInitialRestPoseRecorder`` owned by the unwrapped Arena environment."""

    return get_env(env).object_initial_rest_pose_recorder


def reset_rest_pose_recorder(env, env_ids=None) -> None:
    """Clear recorded initial rest poses for ``env_ids``. Invoked by the progress tracker on env reset."""

    recorder = getattr(get_env(env), "object_initial_rest_pose_recorder", None)
    if recorder is not None:
        recorder.reset(env_ids)


def get_object_initial_rest_state(env, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(position, settled)`` for an object's recorded initial rest pose.

    ``position`` (num_envs, 3) is meaningful only for envs whose ``settled`` mask (num_envs,) is True.
    """

    return get_rest_pose_recorder(env).get(name)


def objects_settled(
    env,
    object_names: list[str],
    lin_vel_threshold: float = 1e-2,
    ang_vel_threshold: float = 5e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """True per env when every object in the env is at rest, records each object's rest pose on first settle.

    An object is at rest when both its linear speed (m/s) and its angular speed (rad/s) are below the
    respective thresholds. The recorded rest poses are readable via ``get_object_initial_rest_state``.
    """

    lin_speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_lin_vel_w(env, name), dim=-1) for name in object_names], dim=0
    )
    ang_speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_ang_vel_w(env, name), dim=-1) for name in object_names], dim=0
    )
    settled = torch.all((lin_speeds < lin_vel_threshold) & (ang_speeds < ang_vel_threshold), dim=0)

    recorder = get_rest_pose_recorder(env)
    for name in object_names:
        recorder.record(name, get_root_pos_w(env, name), settled)

    return select(settled, env_id)
