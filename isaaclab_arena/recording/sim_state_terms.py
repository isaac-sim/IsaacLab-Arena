# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-step capture of restorable simulator state.

The raw-channel recorders answer *what happened*. This answers *where were we*, in a form
``ManagerBasedEnv.reset_to`` accepts — which is what lets a later run resume an episode from the
step at which it went wrong, rather than from a fresh scene.

That distinction is the whole point of recovery-and-correction data (RaC, arXiv:2509.07953): the
high-yield demonstration is not another clean rollout from a fresh initial condition, it is the
rescue from the state the policy actually reached and could not escape. Human interventions are what
make that expensive on hardware; in simulation a restorable state plus a scripted or planned expert
makes it free.

Captured every ``stride`` steps rather than every step: a full scene state is far larger than a
handful of channels, and reset points are only useful at the resolution failures are localised to
(tens of steps), not at 15 Hz.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


def _flatten_state(state: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten Isaac Lab's nested scene state into ``{'articulation/robot/joint_position': tensor}``.

    ``reset_to`` wants the nested form back, so :func:`unflatten_state` is its inverse. Flattening is
    only for storage: HDF5 has no notion of a nested dict, and keeping the path in the key means the
    structure survives a round trip without a schema.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out |= _flatten_state(value, path)
        elif isinstance(value, torch.Tensor):
            out[path] = value
    return out


def unflatten_state(flat: dict[str, Any]) -> dict:
    """Rebuild the nested structure ``reset_to`` expects from flattened keys."""
    nested: dict = {}
    for path, value in flat.items():
        parts = path.split("/")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


class SimStateRecorder(RecorderTerm):
    """Record restorable scene state every ``stride`` steps, plus the final step of the episode."""

    def __init__(self, cfg: SimStateRecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._stride = max(1, int(cfg.stride))
        self._name = cfg.name
        self._layout_written = False

    def record_post_step(self):
        step = int(getattr(self._env, "episode_length_buf", torch.zeros(1))[0].item())
        if step % self._stride:
            return None, None
        # is_relative=True keeps poses in the env frame, so a state captured in one env can be
        # restored into any env index without re-basing by env_origins.
        flat = _flatten_state(self._env.scene.get_state(is_relative=True))
        if not flat:
            return None, None
        self._write_layout_once(flat)
        # One flat vector per step keeps the HDF5 rectangular; the key order goes to the sidecar.
        ordered = [flat[k] for k in sorted(flat)]
        return self._name, torch.cat([t.reshape(t.shape[0], -1) for t in ordered], dim=-1)

    def _write_layout_once(self, flat: dict[str, torch.Tensor]) -> None:
        """Write the key order and widths to a JSON sidecar, so the flat vector can be sliced apart.

        Deliberately NOT emitted as a recorder term. A term that fires on ``record_pre_reset`` also
        fires on the environment's initial reset, which creates a phantom ``demo_0`` holding only the
        layout and shifts every subsequent demo index by one. Every offline consumer here joins
        ``demo_{i}`` to episode row ``i`` positionally, so that would have silently paired every
        episode with the wrong trajectory — the worst kind of failure, because nothing errors.
        """
        if self._layout_written:
            return
        self._layout_written = True
        import json
        import os

        target = os.environ.get("ISAACLAB_DATASET_DIR")
        if not target:
            return
        keys = sorted(flat)
        layout = {k: int(flat[k].reshape(flat[k].shape[0], -1).shape[-1]) for k in keys}
        try:
            pathlib.Path(target, "sim_state_layout.json").write_text(json.dumps(layout, indent=1))
        except OSError:
            pass  # a missing sidecar is recoverable; a crashed rollout is not


@configclass
class SimStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SimStateRecorder
    name: str = "sim_state"
    stride: int = MISSING
    """Steps between captures. Failures localise to tens of steps, so 15 (one second) is ample."""


def make_sim_state_recorder_cfg(stride: int = 15) -> SimStateRecorderCfg:
    return SimStateRecorderCfg(name="sim_state", stride=stride)


# Keys are recorded sorted; this helper documents the contract for offline consumers.
state_key_order: Callable[[dict], list[str]] = lambda state: sorted(_flatten_state(state))
