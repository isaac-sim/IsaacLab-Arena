# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""H1 humanoid embodiment configured for VLN navigation.

Extends the standard H1 embodiment (``isaaclab_arena.embodiments.h1``)
with VLN-specific configuration:
  - Observation layout matching the NaVILA low-level locomotion policy.
  - Velocity command generator for VLM → LL policy integration.
  - VLN-specific event handling.

The observation vector layout is dictated by the pre-trained RSL-RL
locomotion checkpoint and **cannot be reordered**::

    [0:3]   base_lin_vel       (3)
    [3:6]   base_ang_vel       (3)
    [6:9]   projected_gravity  (3)
    [9:12]  velocity_commands  (3)  ← VLM commands injected here
    [12:31] joint_pos_rel      (19)
    [31:50] joint_vel_rel      (19)
    [50:69] last_action        (19)

Camera and lighting are provided by the standard H1 and Matterport
background respectively; this module only adds the observation/command
configuration.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.h1.h1 import (
    H1CameraCfg,
    H1SceneCfg,
    _DEFAULT_H1_CAMERA_OFFSET,
)
from isaaclab_arena.utils.pose import Pose


# ========================================================================== #
# VLN observations                                                           #
# ========================================================================== #
# Layout must match the NaVILA RSL-RL locomotion policy training config.
# See ``env.yaml`` in the training checkpoint directory for the canonical
# definition.  Changing the order will break the pre-trained policy.


@configclass
class H1VlnObservationsCfg:
    """Observation groups for the H1 VLN embodiment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations consumed by the low-level locomotion policy."""

        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        """Duplicate proprio group for the history wrapper (NaVILA compatibility)."""

        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()


# ========================================================================== #
# VLN actions                                                                #
# ========================================================================== #


@configclass
class H1VlnActionCfg:
    """Joint-position action space matching the NaVILA training config."""

    joint_pos: base_mdp.JointPositionActionCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ========================================================================== #
# VLN commands                                                               #
# ========================================================================== #


@configclass
class H1VlnCommandsCfg:
    """Velocity command generator.

    Provides the ``base_velocity`` command term that the proprioceptive
    observation ``velocity_commands`` reads from.  During VLN evaluation
    the actual velocity command is injected by ``VlnPolicy`` directly
    into the observation buffer (indices 9:12).
    """

    base_velocity: base_mdp.UniformVelocityCommandCfg = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# ========================================================================== #
# VLN events                                                                 #
# ========================================================================== #


@configclass
class H1VlnEventCfg:
    """Reset events for the H1 VLN embodiment."""

    reset_robot_joints: EventTerm = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ========================================================================== #
# VLN Embodiment                                                             #
# ========================================================================== #


@register_asset
class H1VlnEmbodiment(EmbodimentBase):
    """H1 humanoid embodiment for Vision-Language Navigation.

    Extends the standard ``H1SceneCfg`` and ``H1CameraCfg`` from
    ``isaaclab_arena.embodiments.h1`` with VLN-specific observations,
    commands, and events.

    Args:
        enable_cameras: Enable camera sensors.
        initial_pose: Robot initial pose.
        camera_offset: First-person camera position on pelvis.
        enable_follow_camera: Add a third-person follow camera for
            visualization.  Default True for VLN.
        follow_camera_offset: Follow camera position on pelvis.
        use_tiled_camera: Use TiledCamera for parallel evaluation.
    """

    name = "h1_vln"

    def __init__(
        self,
        enable_cameras: bool = True,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_H1_CAMERA_OFFSET,
        enable_follow_camera: bool = True,
        follow_camera_offset: Pose | None = None,
        use_tiled_camera: bool = False,
        camera_resolution: int = 512,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)

        # Robot hardware from standard H1
        self.scene_config = H1SceneCfg()

        # Cameras: head (required) + follow (optional).
        # Private attributes must be set BEFORE construction because
        # @configclass __post_init__ reads them at init time.
        cam_cfg = H1CameraCfg.__new__(H1CameraCfg)
        cam_cfg._is_tiled_camera = use_tiled_camera
        cam_cfg._camera_resolution = camera_resolution
        cam_cfg._camera_offset = camera_offset
        cam_cfg._enable_follow_cam = enable_follow_camera
        if follow_camera_offset is not None:
            cam_cfg._follow_camera_offset = follow_camera_offset
        cam_cfg.__init__()
        self.camera_config = cam_cfg

        # VLN-specific configs
        self.action_config = H1VlnActionCfg()
        self.observation_config = H1VlnObservationsCfg()
        self.event_config = H1VlnEventCfg()
        self.command_config = H1VlnCommandsCfg()
