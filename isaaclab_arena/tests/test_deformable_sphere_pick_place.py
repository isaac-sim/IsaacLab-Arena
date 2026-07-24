# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deformable pick-and-place environment (config + Newton end-to-end smoke)."""

import types

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True

_ROBOTIQ_GRIPPER_LABEL_TOKEN = "/Robotiq_2F_85/"
_ROBOTIQ_FINGERTIP_MESH_TOKENS = ("fingertipsstep", "finger4step")
_DROID_GRIPPER_JOINT_NAMES = (
    "finger_joint",
    "left_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
)
_DROID_GRIPPER_CLOSE_DIRECTIONS = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


def _robotiq_shape_labels(shape_labels: list[str]) -> list[str]:
    return [label for label in shape_labels if _ROBOTIQ_GRIPPER_LABEL_TOKEN in label]


def _shape_ids_matching(shape_labels: list[str], *required_tokens: str) -> list[int]:
    return [
        shape_id
        for shape_id, label in enumerate(shape_labels)
        if all(token in label for token in required_tokens) and not label.endswith("_visual")
    ]


def _robotiq_fingertip_shape_ids(shape_labels: list[str]) -> list[int]:
    return [
        shape_id
        for shape_id, label in enumerate(shape_labels)
        if _ROBOTIQ_GRIPPER_LABEL_TOKEN in label
        and any(mesh_name in label for mesh_name in _ROBOTIQ_FINGERTIP_MESH_TOKENS)
        and not label.endswith("_visual")
    ]


def _require_shape_id(shape_labels: list[str], *required_tokens: str) -> int:
    shape_ids = _shape_ids_matching(shape_labels, *required_tokens)
    message = f"Newton imported no collision shape matching {required_tokens}; Robotiq labels were:\n" + "\n".join(
        _robotiq_shape_labels(shape_labels)
    )
    assert shape_ids, message
    return shape_ids[0]


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


def test_deformable_pick_and_place_task_cfg() -> None:
    """The task builds its termination cfg from the shipped pick/destination assets.

    The background is stubbed to its ``object_min_z`` (the only field the task reads): constructing a
    real ``Background`` eagerly opens its remote USD, which requires a running SimulationApp. The real
    ``maple_table_robolab`` background is exercised end-to-end by ``test_deformable_sphere_droid_newton_smoke``.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.deformable_pick_and_place_task import (
        DeformablePickAndPlaceTask,
        DeformablePickAndPlaceTerminationsCfg,
    )

    reg = AssetRegistry()
    sphere = reg.get_asset_by_name("procedural_deformable_sphere")()
    bowl = reg.get_asset_by_name("bowl_ycb_robolab")()
    background = types.SimpleNamespace(name="maple_table_robolab", object_min_z=-0.2)

    task = DeformablePickAndPlaceTask(
        pick_up_object=sphere,
        destination_location=bowl,
        background_scene=background,
    )

    assert task.get_scene_cfg() is None
    assert task.get_events_cfg() is None
    assert isinstance(task.get_termination_cfg(), DeformablePickAndPlaceTerminationsCfg)
    assert isinstance(task.get_metrics()[0], SuccessRateMetric)


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


def _test_deformable_sphere_droid_newton_smoke(simulation_app) -> bool:
    """Boot the shipped deformable env with DROID and check the default Newton VBD path."""
    import torch

    import omni.usd
    import warp as wp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.terms.events import set_deformable_object_pose
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.cli import (
        build_environment_from_cli,
        ensure_environments_registered,
        get_isaaclab_arena_environments_cli_parser,
    )

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name("deformable_sphere_pick_place")

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    args_cli = parser.parse_args([
        "--num_envs",
        "1",
        "deformable_sphere_pick_place",
        "--embodiment",
        "droid_abs_joint_pos",
    ])

    arena_env = build_environment_from_cli(factory_type, args_cli)
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered().unwrapped

    def set_sphere_pose(position_xyz: tuple[float, float, float]) -> None:
        set_deformable_object_pose(
            env,
            env_ids=torch.tensor([0], device=env.device),
            asset_cfg=SceneEntityCfg("procedural_deformable_sphere"),
            pose=Pose(position_xyz=position_xyz),
        )

    def soft_contact_shape_ids() -> set[int]:
        contacts = NewtonCoupledMJWarpVBDManager._contacts
        soft_contact_count = int(wp.to_torch(contacts.soft_contact_count).cpu().item())
        return {
            int(shape_id)
            for shape_id in wp.to_torch(contacts.soft_contact_shape)[:soft_contact_count].detach().cpu().tolist()
        }

    try:
        # Newton VBD backend was actually selected.
        assert env.cfg.scene.replicate_physics is True
        assert env.action_manager.total_action_dim == 8

        env.reset()
        robot = env.scene["robot"]
        asset = env.scene["procedural_deformable_sphere"]
        nodal_before = asset.data.nodal_pos_w.torch.clone()
        assert nodal_before.shape[1] > 0, "deformable has no simulation nodes"
        assert torch.isfinite(nodal_before).all(), "nodal positions not finite after reset"

        hold_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        hold_action[:, :7] = robot.data.joint_pos.torch[:, :7]
        for _ in range(15):
            env.step(hold_action)

        nodal_after = asset.data.nodal_pos_w.torch
        assert torch.isfinite(nodal_after).all(), "nodal positions diverged (non-finite) after stepping"
        # The VBD solver must actually advance the soft body under gravity/contact.
        max_delta = (nodal_after - nodal_before).abs().max().item()
        assert max_delta > 1e-5, f"deformable did not move under Newton stepping (max delta {max_delta})"

        shape_labels = [str(label) for label in NewtonCoupledMJWarpVBDManager._model.shape_label]
        fingertip_shape_ids = _robotiq_fingertip_shape_ids(shape_labels)
        no_fingertip_shapes_message = (
            "Newton imported no Robotiq fingertip collision shapes; labels were:\n"
            + "\n".join(_robotiq_shape_labels(shape_labels))
        )
        assert fingertip_shape_ids, no_fingertip_shapes_message
        right_fingertip_shape_id = _require_shape_id(
            shape_labels,
            _ROBOTIQ_GRIPPER_LABEL_TOKEN,
            "/right_inner_finger/",
            "fingertipsstep",
        )

        stage = omni.usd.get_context().get_stage()
        right_fingertip_label = shape_labels[right_fingertip_shape_id]
        fingertip_prim = stage.GetPrimAtPath(right_fingertip_label)
        assert fingertip_prim.IsValid(), f"right fingertip collision prim is missing: {right_fingertip_label}"
        fingertip_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            fingertip_prim
        )
        fingertip_center = fingertip_bbox.ComputeAlignedBox().GetMidpoint()
        set_sphere_pose((float(fingertip_center[0]), float(fingertip_center[1]), float(fingertip_center[2])))
        nodal_before_contact = asset.data.nodal_pos_w.torch.clone()
        for _ in range(4):
            env.step(hold_action)
        assert (
            right_fingertip_shape_id in soft_contact_shape_ids()
        ), "deformable/right_inner_finger overlap produced no Newton soft contacts with Robotiq fingertip meshes"
        contact_delta = (asset.data.nodal_pos_w.torch - nodal_before_contact).abs().max().item()
        assert contact_delta > 1e-5, f"deformable did not react to fingertip contact (max delta {contact_delta})"

        set_sphere_pose((0.7, 0.3, 0.2))
        for _ in range(2):
            env.step(hold_action)

        joint_ids = [robot.joint_names.index(joint_name) for joint_name in _DROID_GRIPPER_JOINT_NAMES]
        close_targets = torch.tensor(
            [direction * torch.pi / 4 for direction in _DROID_GRIPPER_CLOSE_DIRECTIONS],
            device=env.device,
        )
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        action[:, :7] = robot.data.joint_pos.torch[:, :7]
        action[:, 7] = 1.0
        for _ in range(80):
            env.step(action)
        gripper_close_pos = robot.data.joint_pos.torch[0, joint_ids]
        assert torch.allclose(
            gripper_close_pos, close_targets, atol=0.03
        ), f"DROID gripper did not close to controlled Robotiq targets: {gripper_close_pos}"

        action[:, 7] = 0.0
        for _ in range(80):
            env.step(action)
        gripper_open_pos = robot.data.joint_pos.torch[0, joint_ids]
        assert torch.allclose(
            gripper_open_pos, torch.zeros_like(gripper_open_pos), atol=0.02
        ), f"DROID gripper did not reopen stably: {gripper_open_pos}"
    finally:
        env.close()
    return True


@pytest.mark.with_subprocess
def test_deformable_sphere_droid_newton_smoke() -> None:
    assert run_simulation_app_function(
        _test_deformable_sphere_droid_newton_smoke,
        headless=HEADLESS,
        force_exit_on_success=True,
    )
