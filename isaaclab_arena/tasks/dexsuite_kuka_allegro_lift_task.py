# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dexsuite Kuka Allegro **lift** task for Arena evaluation.

Extends :class:`~isaaclab_arena.tasks.lift_object_task.LiftObjectTask` so the task holds the same
``lift_object`` / ``background_scene`` :class:`~isaaclab_arena.assets.asset.Asset` references as
other lift examples (viewer look-at, future IL/mimic hooks).

Rewards are omitted (evaluation-only); terminations come from Dexsuite plus a position-based
``success`` termination via :func:`~isaaclab_arena.tasks.terminations.lift_object_rl_success`
so :class:`~isaaclab_arena.metrics.success_rate.SuccessRateMetric` works.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.tasks.lift_object_task import LiftObjectTask
from isaaclab_arena.tasks.terminations import lift_object_rl_success


def _build_dexsuite_kuka_lift_cfgs() -> tuple[Any, Any]:
    from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import CommandsCfg

    commands = CommandsCfg()
    commands.object_pose.position_only = True
    terminations = ArenaDexsuiteKukaLiftTerminationsCfg()

    return commands, terminations


@configclass
class ArenaDexsuiteKukaLiftTerminationsCfg(dexsuite.TerminationsCfg):
    """Dexsuite terminations + ``success`` for :class:`~isaaclab_arena.metrics.success_rate.SuccessRateMetric`."""

    success = DoneTerm(
        func=lift_object_rl_success,
        time_out=False,
        params={
            "command_name": "object_pose",
            "position_tolerance": 0.05,
        },
    )


class DexsuiteKukaAllegroLiftTask(LiftObjectTask):
    """Lift task matching Isaac Lab ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` (state observations).

    Reuses :class:`LiftObjectTask` for ``lift_object`` / ``background_scene`` and the default
    look-at-object viewer. MDP pieces come from Dexsuite (not :class:`LiftObjectTaskRL`).
    """

    def __init__(self, lift_object: Asset, background_scene: Asset) -> None:
        # Goal fields from LiftObjectTask are unused for Dexsuite terminations but keep parent API consistent.
        super().__init__(
            lift_object=lift_object,
            background_scene=background_scene,
            episode_length_s=6.0,
            goal_position_delta_xyz=(0.0, 0.0, 0.3),
            goal_position_tolerance=0.05,
        )
        self.task_description = "Dexsuite Kuka Allegro lift (Arena, Newton-ready scene)."

        commands, terminations = _build_dexsuite_kuka_lift_cfgs()
        self._commands_cfg = commands
        self._terminations_cfg = terminations
        self.termination_cfg = self._terminations_cfg

    def get_commands_cfg(self) -> Any:
        return self._commands_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        # Reuse LiftObjectTask's look-at-object framing (same offset as generic lift examples).
        from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

        return get_viewer_cfg_look_at_object(
            lookat_object=self.lift_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        raise NotImplementedError("Dexsuite Kuka Allegro lift mimic is not configured in Arena yet.")

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        # ``DexsuiteReorientEnvCfg.__post_init__`` timing / horizon (sim.dt and physics come from embodiment).
        env_cfg.decimation = 2
        env_cfg.commands.object_pose.resampling_time_range = (2.0, 3.0)
        env_cfg.commands.object_pose.position_only = True
        env_cfg.episode_length_s = 6.0
        env_cfg.is_finite_horizon = False
        return env_cfg
