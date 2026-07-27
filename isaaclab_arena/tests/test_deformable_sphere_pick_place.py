# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for deformable object support and the shared pick-and-place config."""

import types

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def test_deformable_assets_registered() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry

    reg = AssetRegistry()
    assert reg.is_registered("procedural_deformable_sphere")
    assert reg.is_registered("procedural_deformable_cube")


def test_object_hierarchy_reparented() -> None:
    """DeformableObject is a peer of the spawnable objects, not an Object; rigid objects are Spawnable."""
    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectBase, SpawnableObjectBase
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet

    assert issubclass(DeformableObject, ObjectBase)
    assert not issubclass(DeformableObject, SpawnableObjectBase)
    assert not issubclass(DeformableObject, Object)
    assert issubclass(Object, SpawnableObjectBase)
    assert issubclass(ObjectReference, SpawnableObjectBase)
    assert issubclass(RigidObjectSet, Object)


def test_deformable_sphere_cfg_type() -> None:
    """The sphere's object cfg is a soft-body PresetCfg that resolves to a DeformableObjectCfg."""
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab_tasks.utils import PresetCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    sphere.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

    assert sphere.object_type == ObjectType.DEFORMABLE
    assert isinstance(sphere.object_cfg, PresetCfg)

    # Soft-body objects only enumerate soft-body presets; ``default`` maps to the soft default (not
    # PhysX). Both resolve to a DeformableObjectCfg with the initial pose stamped on.
    for backend in ("newton_mjwarp_vbd", "default"):
        cfg = resolve_presets(sphere.object_cfg, selected=(backend,))
        assert isinstance(cfg, DeformableObjectCfg)
        assert cfg.init_state.pos == (0.4, 0.0, 0.1)
    # A nodal reset event is generated for the deformable.
    assert sphere.get_event_cfg()[1] is not None


def test_deformable_spawn_uses_pretet_usd() -> None:
    """The soft-body spawn comes from the committed pre-tetrahedralized TetMesh USD (no pytetwild)."""
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.registries import AssetRegistry

    for asset_name, tet_file in (
        ("procedural_deformable_sphere", "procedural_deformable_sphere_tet.usda"),
        ("procedural_deformable_cube", "procedural_deformable_cube_tet.usda"),
    ):
        asset = AssetRegistry().get_asset_by_name(asset_name)()
        for backend in ("newton_mjwarp_vbd", "default"):
            cfg = resolve_presets(asset.object_cfg, selected=(backend,))
            assert isinstance(cfg.spawn, UsdFileCfg), f"{asset_name}/{backend} is not a UsdFileCfg"
            assert cfg.spawn.usd_path.endswith(tet_file), f"{asset_name}/{backend} not pointing at {tet_file}"


def test_pick_and_place_task_uses_proximity_for_deformable() -> None:
    """PickAndPlaceTask uses generic proximity success when the pickup object has no contact sensor.

    The background is stubbed to its ``object_min_z`` (the only field the task reads): constructing a
    real ``Background`` eagerly opens its remote USD, which requires a running SimulationApp.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask, TerminationsCfg
    from isaaclab_arena.tasks.predicates.spatial import object_is_below_height, objects_in_proximity

    reg = AssetRegistry()
    sphere = reg.get_asset_by_name("procedural_deformable_sphere")()
    bowl = reg.get_asset_by_name("bowl_ycb_robolab")()
    background = types.SimpleNamespace(name="maple_table_robolab", object_min_z=-0.2)

    task = PickAndPlaceTask(
        pick_up_object=sphere,
        destination_location=bowl,
        background_scene=background,
    )

    termination_cfg = task.get_termination_cfg()
    success_predicates = termination_cfg.success.params["predicates"]

    assert task.success_strategy == "proximity"
    assert task.contact_sensor_name is None
    assert task.get_scene_cfg() is None
    assert task.get_events_cfg() is None
    assert isinstance(termination_cfg, TerminationsCfg)
    assert len(success_predicates) == 1
    assert success_predicates[0].func is objects_in_proximity
    assert success_predicates[0].params["velocity_threshold"] == task.velocity_threshold
    assert termination_cfg.object_dropped.func is object_is_below_height
    assert isinstance(task.get_metrics()[0], SuccessRateMetric)


def test_pick_and_place_task_keeps_contact_success_for_contact_objects() -> None:
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.predicates.spatial import object_is_below_height, object_on_destination

    class ContactObject:
        name = "rigid_object"
        prim_path = "{ENV_REGEX_NS}/rigid_object"

        def supports_contact_sensor(self):
            return True

        def get_contact_sensor_cfg(self, contact_against_object=None):
            return ContactSensorCfg(prim_path=self.prim_path)

    pick_up_object = ContactObject()
    destination = types.SimpleNamespace(name="bowl", prim_path="{ENV_REGEX_NS}/bowl")
    background = types.SimpleNamespace(name="table", object_min_z=-0.2)

    task = PickAndPlaceTask(
        pick_up_object=pick_up_object,
        destination_location=destination,
        background_scene=background,
    )
    success_predicates = task.get_termination_cfg().success.params["predicates"]

    assert task.success_strategy == "contact"
    assert task.contact_sensor_name == "contact_sensor_rigid_object"
    assert task.get_scene_cfg() is not None
    assert len(success_predicates) == 1
    assert success_predicates[0].func is object_on_destination
    assert task.get_termination_cfg().object_dropped.func is object_is_below_height


def test_deformable_environment_in_cli_registry() -> None:
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    env_registry = EnvironmentRegistry()
    assert env_registry.is_registered("deformable_sphere_pick_place")
    assert env_registry.get_component_by_name("deformable_sphere_pick_place").name == "deformable_sphere_pick_place"


def test_deformable_physics_backend_selection() -> None:
    """ArenaEnvBuilder defaults deformable scenes to Newton VBD and rejects the rigid ``newton`` preset."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    sphere = AssetRegistry().get_asset_by_name("procedural_deformable_sphere")()
    builder = object.__new__(ArenaEnvBuilder)
    builder.arena_env = types.SimpleNamespace(scene=types.SimpleNamespace(assets={"object": sphere}))

    assert builder._scene_needs_soft_body() is True

    # Non-soft-body presets fail before launch with a clear error.
    for preset in ("default", "newton", "physx"):
        with pytest.raises(NotImplementedError, match="soft-body"):
            builder._select_backend_preset(preset, needs_soft_body=True)

    # No preset on a soft-body scene -> Newton VBD (the PhysX deformable path is unstable).
    assert builder._select_backend_preset(None, needs_soft_body=True) == "newton_mjwarp_vbd"

    # The validated soft-body preset is passed through unchanged.
    assert builder._select_backend_preset("newton_mjwarp_vbd", needs_soft_body=True) == "newton_mjwarp_vbd"

    # Rigid-only scenes with no preset stay on the stock PhysX spawn.
    assert builder._select_backend_preset(None, needs_soft_body=False) is None


def _test_deformable_sphere_newton_smoke(simulation_app) -> bool:
    """Boot a minimal deformable scene and check the default Newton VBD path."""
    import torch

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.terms.events import set_deformable_object_pose
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    sphere = registry.get_asset_by_name("procedural_deformable_sphere")()
    sphere.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.2)))
    ground = registry.get_asset_by_name("ground_plane")()
    embodiment = registry.get_asset_by_name("franka_joint_pos")()
    arena_env = IsaacLabArenaEnvironment(
        name="minimal_deformable_sphere",
        scene=Scene([ground, sphere]),
        embodiment=embodiment,
    )
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False))
    env = builder.make_registered().unwrapped

    def set_sphere_pose(position_xyz: tuple[float, float, float]) -> None:
        set_deformable_object_pose(
            env,
            env_ids=torch.tensor([0], device=env.device),
            asset_cfg=SceneEntityCfg("procedural_deformable_sphere"),
            pose=Pose(position_xyz=position_xyz),
        )

    try:
        # Newton VBD backend was actually selected.
        assert env.cfg.scene.replicate_physics is True

        env.reset()
        asset = env.scene["procedural_deformable_sphere"]
        nodal_before = asset.data.nodal_pos_w.torch.clone()
        assert nodal_before.shape[1] > 0, "deformable has no simulation nodes"
        assert torch.isfinite(nodal_before).all(), "nodal positions not finite after reset"

        hold_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for _ in range(15):
            env.step(hold_action)

        nodal_after = asset.data.nodal_pos_w.torch
        assert torch.isfinite(nodal_after).all(), "nodal positions diverged (non-finite) after stepping"
        # The VBD solver must actually advance the soft body under gravity/contact.
        max_delta = (nodal_after - nodal_before).abs().max().item()
        assert max_delta > 1e-5, f"deformable did not move under Newton stepping (max delta {max_delta})"

        set_sphere_pose((0.1, 0.0, 0.2))
        for _ in range(2):
            env.step(hold_action)
        moved_centroid = asset.data.root_pos_w.torch[0]
        assert torch.isfinite(moved_centroid).all(), "deformable centroid not finite after explicit reset"
        assert torch.allclose(moved_centroid[:2], torch.tensor([0.1, 0.0], device=env.device), atol=0.02)
    finally:
        env.close()
    return True


@pytest.mark.with_subprocess
def test_deformable_sphere_newton_smoke() -> None:
    assert run_simulation_app_function(
        _test_deformable_sphere_newton_smoke,
        headless=HEADLESS,
    )
