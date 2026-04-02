# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kuka + Allegro embodiment for Dexsuite-style manipulation.

Aligned with
:class:`isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg.KukaAllegroMixinCfg`:
robot, fingertip contact sensors, relative joint actions, state observations,
PhysX vs Newton event presets, and Dexsuite simulation rates.

Scene geometry (object, table, ground, lights) is supplied by the Arena :class:`~isaaclab_arena.scene.scene.Scene`;
this embodiment only adds the robot and sensors.
"""

from __future__ import annotations

from typing import Any, Literal

from isaaclab.assets import ArticulationCfg
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
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.pose import Pose

FINGERTIP_LIST = kuka_dexsuite_cfg.FINGERTIP_LIST


# ---------------------------------------------------------------------------
# Observation cfgs (Arena-local): Isaac Lab's ``StateObservationCfg`` / camera obs use ``super()`` in
# ``__post_init__``. Arena's :func:`~isaaclab_arena.utils.configclass.combine_configclass_instances`
# calls those ``__post_init__`` hooks with ``self`` equal to the *merged* config type, which is not a
# subclass of ``StateObservationCfg``, so ``super()`` raises. We duplicate upstream layout with explicit
# ``Parent.__post_init__(self)`` calls instead (same as ``camera_cfg.StateObservationCfg`` et al.).
# ---------------------------------------------------------------------------


@configclass
class ArenaDexsuiteKukaStateObservationCfg(dexsuite.ObservationsCfg):
    """State observations matching ``camera_cfg.StateObservationCfg``; merge-safe ``__post_init__``."""

    def __post_init__(self) -> None:
        dexsuite.ObservationsCfg.__post_init__(self)
        self.proprio.contact = ObsTerm(
            func=dexsuite_mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in FINGERTIP_LIST]},
            clip=(-20.0, 20.0),
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]


@configclass
class DexsuiteKukaAllegroEmbodimentSceneCfg:
    """Robot and fingertip contact sensors (Dexsuite naming).

    .. note::
        Do not set :attr:`replicate_physics` here. It belongs on :class:`~isaaclab.scene.InteractiveSceneCfg`;
        merging both into one Arena scene class triggers a type clash across Isaac Lab versions.
        :meth:`KukaAllegroDexsuiteEmbodiment.modify_env_cfg` sets ``env_cfg.scene.replicate_physics = True``
        to match Dexsuite.
    """

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


@register_asset
class KukaAllegroDexsuiteEmbodiment(EmbodimentBase):
    """Kuka Allegro for Dexsuite tasks: joint-space actions, contact-rich proprioception."""

    name = "kuka_allegro_dexsuite"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        physics_preset: Literal["physx", "newton"] = "physx",
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = True,
        arm_mode: ArmMode | None = None,
    ):
        # Default ``True``: Dexsuite ``PolicyCfg`` / ``ProprioObsCfg`` use ``concatenate_terms=True`` in Isaac Lab;
        # RSL-RL MLPs require flat 1D-per-env vectors. ``False`` yields invalid shapes (e.g. ``torch.Size([1])``).
        super().__init__(False, initial_pose, concatenate_observation_terms, arm_mode)
        self.physics_preset = physics_preset

        self.scene_config = DexsuiteKukaAllegroEmbodimentSceneCfg()
        self.action_config = kuka_dexsuite_cfg.KukaAllegroRelJointPosActionCfg()
        self.observation_config = ArenaDexsuiteKukaStateObservationCfg()
        self._apply_concatenate_observation_terms(self.observation_config)

        # ``PresetCfg`` subclasses (e.g. ``KukaAllegroEventCfg``) store presets as *instance* fields, not
        # class attributes — ``KukaAllegroEventCfg.newton`` raises ``AttributeError``.
        _event_presets = kuka_dexsuite_cfg.KukaAllegroEventCfg()
        if physics_preset == "newton":
            self.event_config = getattr(_event_presets, "newton", None)
            if self.event_config is None:
                from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as _dexsuite

                self.event_config = _dexsuite.EventCfg()
        else:
            self.event_config = _event_presets.default

        self.reward_config = None
        self.command_config = None
        self.termination_cfg = None
        self.curriculum_config = None
        self.mimic_env = None
        self.camera_config = None

    def _apply_concatenate_observation_terms(self, obs_cfg: Any) -> None:
        for field_name in ("policy", "proprio", "perception"):
            if hasattr(obs_cfg, field_name):
                grp = getattr(obs_cfg, field_name)
                if hasattr(grp, "concatenate_terms"):
                    grp.concatenate_terms = self.concatenate_observation_terms

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        physics_cfg = kuka_dexsuite_cfg.KukaAllegroPhysicsCfg()
        env_cfg.sim.physics = getattr(physics_cfg, self.physics_preset, physics_cfg.default)
        env_cfg.sim.dt = 1 / 120
        env_cfg.decimation = 2
        # Dexsuite uses replicated physics; Arena's builder defaults InteractiveSceneCfg to False.
        if hasattr(env_cfg, "scene") and env_cfg.scene is not None:
            env_cfg.scene.replicate_physics = True
        return env_cfg

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "palm_link"

    def get_command_body_name(self) -> str:
        # Dexsuite object_pose command uses asset frame on the robot, not a named body.
        return ""
