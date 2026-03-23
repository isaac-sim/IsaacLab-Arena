# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Success-rate metric for Dexsuite lift-style MDPs.

Dexsuite uses :class:`success_reward` (sticky ``succeeded`` on the reward term) rather than a
``success`` *termination*. :class:`~isaaclab_arena.metrics.success_rate.SuccessRecorder` therefore
cannot be used; this module records from ``reward_manager.get_term_cfg("success").func.succeeded``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.success_rate import SuccessRateMetric


class DexsuiteLiftSuccessRecorder(RecorderTerm):
    """Records whether the episode achieved Dexsuite success (sticky reward term), pre-reset."""

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.first_reset = True

    def record_pre_reset(self, env_ids: Sequence[int]):
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        if "success" not in self._env.reward_manager.active_terms:
            raise RuntimeError(
                "DexsuiteLiftSuccessRecorder requires a reward term named 'success' "
                "(Dexsuite ``success_reward``)."
            )
        term_cfg = self._env.reward_manager.get_term_cfg("success")
        func = term_cfg.func
        if not hasattr(func, "succeeded"):
            raise TypeError(
                f"Reward term 'success' must be a stateful term with `.succeeded` (e.g. Dexsuite "
                f"`success_reward`), got {type(func)}."
            )
        success_results = func.succeeded[env_ids]
        return self.name, success_results


@configclass
class DexsuiteLiftSuccessRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = DexsuiteLiftSuccessRecorder
    name: str = "dexsuite_lift_success"


class DexsuiteLiftSuccessRateMetric(SuccessRateMetric):
    """Same aggregation as :class:`SuccessRateMetric`, sourcing labels from Dexsuite success reward."""

    recorder_term_name: str = "dexsuite_lift_success"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return DexsuiteLiftSuccessRecorderCfg(name=self.recorder_term_name)
