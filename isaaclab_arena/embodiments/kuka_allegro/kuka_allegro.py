# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kuka + Allegro embodiment for dexterous manipulation.

Robot, fingertip contact sensors, relative joint actions, state observations.

Scene geometry (object, table, ground, lights) is supplied by the Arena `isaaclab_arena.scene.scene.Scene`;
this embodiment only adds the robot and sensors.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_tasks.core.lift import lift_env_cfg as lift
from isaaclab_tasks.core.lift import mdp as lift_mdp
from isaaclab_tasks.core.lift.config.kuka_allegro import kuka_allegro_env_cfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

FINGERTIP_LIST = kuka_allegro_env_cfg.FINGERTIP_LIST


@configclass
class KukaAllegroStateObservationCfg(lift.ObservationsCfg):
    """State observations with fingertip contact; merge-safe ``__post_init__``."""

    def __post_init__(self) -> None:
        lift.ObservationsCfg.__post_init__(self)
        self.proprio.contact = ObsTerm(
            func=lift_mdp.fingers_contact_force_b,
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

    registry_name = "kuka_allegro"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = True,
        arm_mode: ArmMode | None = None,
    ):
        """Create the Kuka Allegro embodiment with relative joint position actions and proprioceptive observations.

        Args:
            enable_cameras: Must be false because this embodiment has no cameras.
            initial_pose: Pose of the robot base in the environment.
            concatenate_observation_terms: Whether policy observation terms are concatenated into one tensor.
            arm_mode: Which arm(s) the embodiment controls.
        """
        assert not enable_cameras, "The kuka_allegro embodiment carries no cameras"
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)

        self.scene_config = KukaAllegroSceneCfg()
        self.action_config = kuka_allegro_env_cfg.KukaAllegroRelJointPosActionCfg()
        self.observation_config = KukaAllegroStateObservationCfg()
        self.event_config = KukaAllegroEventCfg()

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        """Return the name of the end-effector frame, which is the palm link.

        Args:
            arm_mode: Arm mode to look up the frame for. Not used; there is a single end-effector frame.

        Returns:
            The end-effector frame name.
        """
        return "palm_link"

    def get_command_body_name(self) -> str:
        """Return an empty string, since the actions are joint-space and command no body."""
        return ""
