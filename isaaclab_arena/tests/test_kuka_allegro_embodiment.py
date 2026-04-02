# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`KukaAllegroDexsuiteEmbodiment` (no simulation)."""

import pytest

pytest.importorskip(
    "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg"
)

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.embodiments.kuka_allegro.kuka_allegro import (
    DexsuiteKukaAllegroEmbodimentSceneCfg,
    KukaAllegroDexsuiteCameraSceneCfg,
    KukaAllegroDexsuiteEmbodiment,
)


def test_kuka_allegro_dexsuite_registered_in_asset_registry() -> None:
    reg = AssetRegistry()
    assert reg.is_registered("kuka_allegro")
    cls = reg.get_asset_by_name("kuka_allegro")
    assert cls is KukaAllegroDexsuiteEmbodiment


def test_scene_cfg_has_robot_and_fingertip_contact_sensors() -> None:
    cfg = DexsuiteKukaAllegroEmbodimentSceneCfg()
    assert cfg.robot is not None
    for link in ("index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"):
        attr = f"{link}_object_s"
        assert hasattr(cfg, attr), f"missing sensor {attr}"
        sensor = getattr(cfg, attr)
        assert "{ENV_REGEX_NS}/Object" in sensor.filter_prim_paths_expr


def test_embodiment_default_action_and_observation() -> None:
    emb = KukaAllegroDexsuiteEmbodiment()
    assert emb.physics_preset == "physx"
    assert emb.enable_cameras is False
    assert emb.duo_cameras is False
    assert emb.concatenate_observation_terms is True
    assert emb.observation_config.policy.concatenate_terms is True
    assert emb.action_config.action.scale == 0.1
    assert emb.action_config.action.joint_names == [".*"]
    assert emb.action_config.action.asset_name == "robot"
    # State observation extends dexsuite ObservationsCfg with fingertip contact term
    assert hasattr(emb.observation_config, "proprio")
    assert hasattr(emb.observation_config.proprio, "contact")


def test_embodiment_physx_vs_newton_events() -> None:
    physx_emb = KukaAllegroDexsuiteEmbodiment(physics_preset="physx")
    newton_emb = KukaAllegroDexsuiteEmbodiment(physics_preset="newton")
    assert physx_emb.event_config is not newton_emb.event_config


def test_embodiment_enable_cameras_sets_camera_scene_and_observations() -> None:
    single = KukaAllegroDexsuiteEmbodiment(enable_cameras=True, duo_cameras=False)
    assert single.camera_config is not None
    assert isinstance(single.camera_config, KukaAllegroDexsuiteCameraSceneCfg)
    assert single.camera_config.base_camera is not None
    assert single.camera_config.wrist_camera is None
    assert hasattr(single.observation_config, "base_image")

    duo = KukaAllegroDexsuiteEmbodiment(enable_cameras=True, duo_cameras=True)
    assert duo.camera_config.wrist_camera is not None
    assert hasattr(duo.observation_config, "wrist_image")


def test_get_scene_cfg_includes_cameras_when_enabled() -> None:
    emb = KukaAllegroDexsuiteEmbodiment(enable_cameras=True, duo_cameras=True)
    scene_cfg = emb.get_scene_cfg()
    assert hasattr(scene_cfg, "base_camera")
    assert scene_cfg.base_camera is not None
    assert hasattr(scene_cfg, "wrist_camera")
    assert scene_cfg.wrist_camera is not None


def test_modify_env_cfg_sets_dexsuite_timestep_and_physics() -> None:
    """Duck-typed env cfg: ``sim``, ``decimation``, and optional ``scene.replicate_physics``."""
    from isaaclab_physx.physics import PhysxCfg

    class _Sim:
        def __init__(self) -> None:
            self.dt = 0.01
            self.physics = None

    class _Scene:
        def __init__(self) -> None:
            self.replicate_physics = False

    class _EnvCfg:
        def __init__(self) -> None:
            self.sim = _Sim()
            self.decimation = 4
            self.scene = _Scene()

    emb = KukaAllegroDexsuiteEmbodiment(physics_preset="physx")
    cfg = _EnvCfg()
    out = emb.modify_env_cfg(cfg)  # type: ignore[arg-type]
    assert out.decimation == 2
    assert out.sim.dt == pytest.approx(1 / 120)
    assert isinstance(out.sim.physics, PhysxCfg)
    assert out.scene.replicate_physics is True
