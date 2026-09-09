# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Forward IsaacLab RecorderManager callbacks to externally-supplied functions.

CallbackRecorderTerm carries no domain knowledge of its own. It exists so callers can
get correctly-timed per-step and pre-reset hooks (the same timing IsaacLab's own
RecorderManager already uses for HDF5 demo/mimic export) without writing a new
RecorderTerm subclass for each use case.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@dataclass(frozen=True)
class CallbackRecorderTermHandlers:
    """Plain lifecycle callbacks a CallbackRecorderTerm forwards to. All optional."""

    on_post_step: Callable[[ManagerBasedEnv], None] | None = None
    """Called once per step, for every env, before any reset that step."""

    on_pre_reset: Callable[[ManagerBasedEnv, Sequence[int]], None] | None = None
    """Called once per resetting batch, before the reset is effective."""

    on_close: Callable[[str | None], None] | None = None
    """Called when the owning RecorderManager is closed."""


class CallbackRecorderTerm(RecorderTerm):
    """Forward RecorderManager's per-step and pre-reset callbacks to configured handlers.

    Writes no data of its own (always returns (None, None)) -- it exists purely to get
    correctly-timed callbacks into arbitrary code, not to contribute to the exported
    HDF5 dataset.
    """

    def __init__(self, cfg: CallbackRecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._handlers = cfg.build_handlers(env)

    def record_post_step(self):
        if self._handlers.on_post_step is not None:
            self._handlers.on_post_step(self._env)
        return None, None

    def record_pre_reset(self, env_ids):
        if self._handlers.on_pre_reset is not None:
            self._handlers.on_pre_reset(self._env, env_ids)
        return None, None

    def close(self, file_path):
        if self._handlers.on_close is not None:
            self._handlers.on_close(file_path)


@configclass
class CallbackRecorderTermCfg(RecorderTermCfg):
    """Configuration for a CallbackRecorderTerm."""

    class_type: type[RecorderTerm] = CallbackRecorderTerm

    build_handlers: Callable[[ManagerBasedEnv], CallbackRecorderTermHandlers] = None
    """Called once, at term construction time, with the live env, to obtain the handlers.

    Deferred like this (rather than passing already-built handlers) because RecorderTerm
    construction happens during env.load_managers() -- callers that need to build something
    from the env itself (e.g. a datagen collector) cannot do so before this point.
    """
