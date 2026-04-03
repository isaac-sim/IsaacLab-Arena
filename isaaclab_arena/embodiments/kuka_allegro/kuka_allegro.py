# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kuka + Allegro embodiment for dexterous manipulation.

Robot, fingertip contact sensors, relative joint actions, state observations.

Scene geometry (object, table, ground, lights) is supplied by the Arena :class:`~isaaclab_arena.scene.scene.Scene`;
this embodiment only adds the robot and sensors.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp as dexsuite_mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro import (
    dexsuite_kuka_allegro_env_cfg as kuka_dexsuite_cfg,
)

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

FINGERTIP_LIST = kuka_dexsuite_cfg.FINGERTIP_LIST


@configclass
class KukaAllegroStateObservationCfg(dexsuite.ObservationsCfg):
    """State observations with fingertip contact; merge-safe ``__post_init__``."""

    def __post_init__(self) -> None:
        dexsuite.ObservationsCfg.__post_init__(self)
        self.proprio.contact = ObsTerm(
            func=dexsuite_mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in FINGERTIP_LIST]},
            clip=(-20.0, 20.0),
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]


@configclass
class KukaAllegroSceneCfg:
    """Robot and fingertip contact sensors."""

    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    index_link_3_object_s: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/index_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    middle_link_3_object_s: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/middle_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    ring_link_3_object_s: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/ring_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    thumb_link_3_object_s: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/thumb_link_3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class KukaAllegroEventCfg:
    """Reset event: randomize joint positions on episode reset."""

    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )


@register_asset
class KukaAllegroEmbodiment(EmbodimentBase):
    """Kuka Allegro: joint-space actions, contact-rich proprioception."""

    name = "kuka_allegro"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = True,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(False, initial_pose, concatenate_observation_terms, arm_mode)

        self.scene_config = KukaAllegroSceneCfg()
        self.action_config = kuka_dexsuite_cfg.KukaAllegroRelJointPosActionCfg()
        self.observation_config = KukaAllegroStateObservationCfg()
        self.event_config = KukaAllegroEventCfg()

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "palm_link"

    def get_command_body_name(self) -> str:
        return ""
