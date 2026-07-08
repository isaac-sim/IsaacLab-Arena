# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Object settling predicate and cross-predicate object settled state recording.

The ``object_settled`` predicate reports when all objects in the env come to a rest at the beginning of an episode.
When an object settles, its resting position is recorded. Downstream predicates can read it with ``get_object_settled_state``.
"""


from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_env,
    get_root_ang_vel_w,
    get_root_lin_vel_w,
    get_root_pos_w,
    select,
)

_SETTLED_OBJ_POS_ATTR = "_settled_object_positions"


def objects_settled(
    env,
    object_names: list[str],
    lin_vel_threshold: float = 1e-2,
    ang_vel_threshold: float = 1e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """True per env when every object in the env is at rest, records each object's position on first settle.

    An object is at rest when both its linear speed (m/s) and its angular speed (rad/s) are below the
    respective thresholds.
    """

    lin_speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_lin_vel_w(env, name), dim=-1) for name in object_names], dim=0
    )
    ang_speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_ang_vel_w(env, name), dim=-1) for name in object_names], dim=0
    )
    settled = torch.all((lin_speeds < lin_vel_threshold) & (ang_speeds < ang_vel_threshold), dim=0)
    for name in object_names:
        _record_object_settled_pos(env, name, settled)

    return select(settled, env_id)


def get_object_settled_state(env, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the settled state of an object.

    The settled state is a tuple of two tensors, the settled pos and the settled mask.
    The settled pos (num_envs, 3) is meaningful when the settled mask (num_envs,) is True.
    """

    entry = _get_entry(env, name)
    return entry["pos"], entry["settled"]


def _get_entry(env, name: str) -> dict[str, torch.Tensor]:
    """Get or create the ``{pos, settled}`` record for one object, cached on the (unwrapped) env."""

    env = get_env(env)
    store = getattr(env, _SETTLED_OBJ_POS_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _SETTLED_OBJ_POS_ATTR, store)
    if name not in store:
        store[name] = {
            "pos": torch.full((env.num_envs, 3), float("nan"), device=env.device),
            "settled": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
    return store[name]


def _record_object_settled_pos(env, name: str, settled: torch.Tensor) -> None:
    """Record ``name``'s position for envs that just settled and weren't recorded yet."""

    entry = _get_entry(env, name)
    new = settled & ~entry["settled"]
    if bool(new.any()):
        entry["pos"] = torch.where(new.unsqueeze(-1), get_root_pos_w(env, name), entry["pos"])
        entry["settled"] = entry["settled"] | new


def reset_settled(env, env_ids=None) -> None:
    """Clear recorded settle data for ``env_ids`` (all envs if None). Runs on env reset."""

    store = getattr(get_env(env), _SETTLED_OBJ_POS_ATTR, None)
    if store is None:
        return
    ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=get_env(env).device)
    for entry in store.values():
        entry["settled"][ids] = False
        entry["pos"][ids] = float("nan")


@configclass
class ObjectSettlingResetEventCfg:
    """Clears recorded settle data as envs reset."""

    reset_settled: EventTermCfg = EventTermCfg(func=reset_settled, mode="reset")


def make_object_settling_reset_event_cfg() -> ObjectSettlingResetEventCfg:
    """Reset-event cfg that clears recorded settle data on env reset."""

    return ObjectSettlingResetEventCfg()
