# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dexsuite Kuka Allegro **lift** MDP: commands, rewards, success signal, terminations, curriculum.

Mirrors :class:`isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg.DexsuiteKukaAllegroLiftEnvCfg`
(Kuka-specific reward params + lift: no orientation reward / position-only goal / ``success`` without orientation).
"""

from __future__ import annotations

from typing import Any

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

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase


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


class DexsuiteKukaAllegroLiftTask(TaskBase):
    """Lift task matching Isaac Lab ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` (state observations)."""

    def __init__(self) -> None:
        super().__init__(episode_length_s=6.0, task_description="Dexsuite Kuka Allegro lift (Arena, Newton-ready scene).")
        commands, rewards, terminations, curriculum = _build_dexsuite_kuka_lift_cfgs()
        self._commands_cfg = commands
        self._rewards_cfg = rewards
        self._terminations_cfg = terminations
        self._curriculum_cfg = curriculum
        self._viewer_cfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")

    def get_scene_cfg(self) -> Any:
        # Scene layout comes from Arena assets; ``replicate_physics`` is applied in the Kuka Dexsuite embodiment's
        # :meth:`modify_env_cfg` (not on the embodiment scene cfg, to avoid merging clashes with InteractiveSceneCfg).
        return None

    def get_commands_cfg(self) -> Any:
        return self._commands_cfg

    def get_rewards_cfg(self) -> Any:
        return self._rewards_cfg

    def get_termination_cfg(self) -> Any:
        return self._terminations_cfg

    def get_curriculum_cfg(self) -> Any:
        return self._curriculum_cfg

    def get_events_cfg(self) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        # Dexsuite uses reward-term success, not a ``success`` termination; skip SuccessRate recorder.
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return self._viewer_cfg

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
