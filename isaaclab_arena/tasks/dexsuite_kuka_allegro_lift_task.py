# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dexsuite Kuka Allegro **lift** MDP: commands, rewards, success signal, terminations, curriculum.

Extends :class:`~isaaclab_arena.tasks.lift_object_task.LiftObjectTask` so the task holds the same
``lift_object`` / ``background_scene`` :class:`~isaaclab_arena.assets.asset.Asset` references as
other lift examples (viewer look-at, future IL/mimic hooks). The **MDP** still mirrors Isaac Lab
:class:`~isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg.DexsuiteKukaAllegroLiftEnvCfg`
(Dexsuite commands/rewards/terminations/curriculum), not Arena's generic
:class:`~isaaclab_arena.tasks.lift_object_task.LiftObjectRewardCfg` stack.

After construction, parent's IL-style :attr:`~isaaclab_arena.tasks.lift_object_task.LiftObjectTask.termination_cfg`
is replaced with Dexsuite :class:`TerminationsCfg`; :meth:`get_commands_cfg` / :meth:`get_rewards_cfg` /
:meth:`get_curriculum_cfg` return Dexsuite configs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp as dexsuite_mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
    FINGER_SENSORS,
    THUMB_SENSOR,
)

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.lift_object_task import LiftObjectTask


@configclass
class ArenaDexsuiteKukaReorientRewardCfg(dexsuite.RewardsCfg):
    """Same as ``KukaAllegroReorientRewardCfg`` but merge-safe: no ``super()`` in ``__post_init__``."""

    good_finger_contact = RewTerm(
        func=dexsuite_mdp.contacts,
        weight=0.5,
        params={"threshold": 0.1, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=dexsuite_mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": FINGER_SENSORS + [THUMB_SENSOR],
        },
    )

    def __post_init__(self) -> None:
        dexsuite.RewardsCfg.__post_init__(self)
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


def _build_dexsuite_kuka_lift_cfgs() -> tuple[Any, Any, Any, Any]:
    from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg
    from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import CommandsCfg, TerminationsCfg

    commands = CommandsCfg()
    rewards = ArenaDexsuiteKukaReorientRewardCfg()
    terminations = TerminationsCfg()
    curriculum = CurriculumCfg()

    # ``DexsuiteLiftEnvCfg.__post_init__``
    rewards.orientation_tracking = None
    commands.object_pose.position_only = True
    rewards.success.params["rot_std"] = None

    return commands, rewards, terminations, curriculum


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

        commands, rewards, terminations, curriculum = _build_dexsuite_kuka_lift_cfgs()
        self._commands_cfg = commands
        self._rewards_cfg = rewards
        self._terminations_cfg = terminations
        self._curriculum_cfg = curriculum
        # Replace parent's IL terminations with Dexsuite MDP terminations.
        self.termination_cfg = self._terminations_cfg

    def get_commands_cfg(self) -> Any:
        return self._commands_cfg

    def get_rewards_cfg(self) -> Any:
        return self._rewards_cfg

    def get_curriculum_cfg(self) -> Any:
        return self._curriculum_cfg

    def get_metrics(self) -> list[MetricBase]:
        # Dexsuite uses reward-term success, not a ``success`` termination; skip SuccessRate recorder.
        return []

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
