# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`KukaAllegroEmbodiment` (no simulation)."""

import pytest

pytest.importorskip(
    "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg"
)

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.embodiments.kuka_allegro.kuka_allegro import KukaAllegroEmbodiment, KukaAllegroSceneCfg


def test_kuka_allegro_registered_in_asset_registry() -> None:
    reg = AssetRegistry()
    assert reg.is_registered("kuka_allegro")
    cls = reg.get_asset_by_name("kuka_allegro")
    assert cls is KukaAllegroEmbodiment


def test_scene_cfg_has_robot_and_fingertip_contact_sensors() -> None:
    cfg = KukaAllegroSceneCfg()
    assert cfg.robot is not None
    for link in ("index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"):
        attr = f"{link}_object_s"
        assert hasattr(cfg, attr), f"missing sensor {attr}"
        sensor = getattr(cfg, attr)
        assert "{ENV_REGEX_NS}/Object" in sensor.filter_prim_paths_expr


def test_embodiment_default_action_and_observation() -> None:
    emb = KukaAllegroEmbodiment()
    assert emb.concatenate_observation_terms is True
    assert emb.observation_config.policy.concatenate_terms is True
    assert emb.action_config.action.scale == 0.1
    assert emb.action_config.action.joint_names == [".*"]
    assert emb.action_config.action.asset_name == "robot"
    assert hasattr(emb.observation_config, "proprio")
    assert hasattr(emb.observation_config.proprio, "contact")


def test_embodiment_has_reset_event() -> None:
    from isaaclab_arena.embodiments.kuka_allegro.kuka_allegro import KukaAllegroEventCfg

    emb = KukaAllegroEmbodiment()
    assert isinstance(emb.event_config, KukaAllegroEventCfg)
    assert hasattr(emb.event_config, "reset_robot_joints")
    assert emb.event_config.reset_robot_joints.mode == "reset"
