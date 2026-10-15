# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import tqdm
import traceback
from itertools import product
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
NUM_ENVS = 10
# Kept lower than NUM_ENVS because each env renders three 720p cameras. Must stay > 1 so the
# object set still produces a heterogeneous clone plan.
NUM_ENVS_WITH_CAMERAS = 4
NUM_STEPS_WITH_CAMERAS = 2
OBJECT_SET_1_PRIM_PATH = "/World/envs/env_.*/ObjectSet_1"
OBJECT_SET_2_PRIM_PATH = "/World/envs/env_.*/ObjectSet_2"
OBJECT_SET_JUG_PRIM_PATH = "/World/envs/env_.*/ObjectSet_Jug"
OBJECT_SET_BOTTLES_PRIM_PATH = "/World/envs/env_.*/ObjectSet_Bottles"


def _make_two_primitive_members():
    """Return two rigid cuboid members of different sizes and their bounding boxes."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    can_a = Object(
        name="can_a",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg()),
    )
    can_b = Object(
        name="can_b",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.3), rigid_props=sim_utils.RigidBodyPropertiesCfg()),
    )
    return can_a, can_b, can_a.get_bounding_box(), can_b.get_bounding_box()


def _patch_contact_body_path(contact_body_path="/rigid"):
    """Patch contact-body discovery, so members with placeholder USD paths need no file."""
    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry

    return patch.object(SpawnCfgGeometry, "get_contact_body_path", autospec=True, return_value=contact_body_path)


def _write_rigid_cube_usd(usd_path, body_path="/Object", default_path=None):
    """Write a USD file with one rigid body at body_path that holds a 0.1 m collision cube.

    The default prim is default_path, or the first prefix of body_path when it is None.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(usd_path))
    body_prefixes = Sdf.Path(body_path).GetPrefixes()
    for prefix in body_prefixes:
        UsdGeom.Xform.Define(stage, prefix)
    if default_path is None:
        default_path = body_prefixes[0]
    stage.SetDefaultPrim(stage.GetPrimAtPath(default_path))
    body = stage.GetPrimAtPath(body_path)
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    cube = UsdGeom.Cube.Define(stage, f"{body_path}/Cube")
    cube.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    return str(usd_path)


def _test_object_set_assignment_drives_bounds_and_contact_paths(simulation_app):
    """A set's spawn entry indices drive its per-env bounds, and its contact paths come from its members."""
    import torch

    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry

    can_a, can_b, bbox_a, bbox_b = _make_two_primitive_members()
    obj_set = RigidObjectSet(name="cans", objects=[can_a, can_b], random_choice=True)
    destination_set = RigidObjectSet(name="bins", objects=[can_b, can_a])
    spawn_cfg = obj_set.object_cfg.spawn
    spawn_cfg.asset_indices_by_env = (1, 0, 1, 1)
    # Isaac Lab's own random choice stays off.
    assert spawn_cfg.random_choice is False

    per_env_bbox = obj_set.get_bounding_box_per_env(num_envs=4)
    member_bounding_boxes = (bbox_a, bbox_b)
    expected_min_points = torch.cat([member_bounding_boxes[index].min_point for index in (1, 0, 1, 1)])
    expected_max_points = torch.cat([member_bounding_boxes[index].max_point for index in (1, 0, 1, 1)])
    assert torch.allclose(per_env_bbox.min_point, expected_min_points)
    assert torch.allclose(per_env_bbox.max_point, expected_max_points)

    with _patch_contact_body_path():
        contact_sensor_cfg = obj_set.get_contact_sensor_cfg(contact_against_object=destination_set)
    assert contact_sensor_cfg.prim_path == f"{obj_set.prim_path}/rigid"
    assert contact_sensor_cfg.filter_prim_paths_expr == [f"{destination_set.prim_path}/rigid"]

    def contact_body_path_by_member(geometry):
        return "/rigid" if geometry.spawn_cfg.size == (0.1, 0.1, 0.2) else "/other_rigid"

    with patch.object(
        SpawnCfgGeometry, "get_contact_body_path", autospec=True, side_effect=contact_body_path_by_member
    ):
        with pytest.raises(AssertionError, match="same contact-sensor prim path"):
            obj_set.get_contact_sensor_prim_path()
    return True


def _test_two_object_sets_resolve_normalized_usd_contact_paths(simulation_app, tmp_path):
    """Task sensor and filter paths must resolve for every pair of members, with and without moved bodies."""
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    def make_member(name, default_path, body_path):
        usd_path = _write_rigid_cube_usd(tmp_path / f"{name}.usda", body_path, default_path=default_path)
        return Object(name=name, object_type=ObjectType.RIGID, usd_path=usd_path)

    pickup_members = [
        make_member("pickup_root", "/Fruit", "/Fruit"),
        make_member("pickup_nested", "/Outer/Asset", "/Outer/Asset"),
    ]
    destination_members = [
        make_member("bin_shallow", "/Bin", "/Bin/Body"),
        make_member("bin_deep", "/Outer/Bin", "/Outer/Bin/Geometry/Body"),
    ]
    with patch("isaaclab_arena.utils.usd.rigid_body_path_alignment.get_arena_asset_cache_dir", return_value=tmp_path):
        pickup = RigidObjectSet(name="pickup", objects=pickup_members)
        destination = RigidObjectSet(name="destination", objects=destination_members)

    pickup_entry_paths = [entry.usd_path for entry in pickup.object_cfg.spawn.assets_cfg]
    destination_entry_paths = [entry.usd_path for entry in destination.object_cfg.spawn.assets_cfg]
    # Pickup members have their bodies at the default prim, so they spawn their own files.
    assert pickup_entry_paths == [obj.usd_path for obj in pickup_members]
    # Destination bodies sit at different depths, so they spawn copies with the body moved to a common path.
    assert {obj.usd_path for obj in destination_members}.isdisjoint(destination_entry_paths)
    task = PickAndPlaceTask(pick_up_object=pickup, destination_location=destination, background_scene=destination)
    sensor_cfg = getattr(task.get_scene_cfg(), task.contact_sensor_name)
    assert sensor_cfg.prim_path == pickup.prim_path
    assert sensor_cfg.filter_prim_paths_expr == [f"{destination.prim_path}/rigid_body"]

    stage = Usd.Stage.CreateInMemory()
    for env_idx, (pickup_path, destination_path) in enumerate(product(pickup_entry_paths, destination_entry_paths)):
        env_path = f"/World/envs/env_{env_idx}"
        for obj_set, member_path in ((pickup, pickup_path), (destination, destination_path)):
            holder = stage.DefinePrim(obj_set.prim_path.replace("{ENV_REGEX_NS}", env_path), "Xform")
            holder.GetReferences().AddReference(member_path)
        for path_expr in (sensor_cfg.prim_path, *sensor_cfg.filter_prim_paths_expr):
            body = stage.GetPrimAtPath(path_expr.replace("{ENV_REGEX_NS}", env_path))
            assert body.IsValid() and body.HasAPI(UsdPhysics.RigidBodyAPI), path_expr
            assert body.GetChild("Cube").HasAPI(UsdPhysics.CollisionAPI), path_expr
    return True


def _test_object_set_spawns_members_from_their_own_spawn_cfgs(simulation_app, tmp_path):
    """The set spawns copies of each member's own spawn configuration, including primitive members."""
    import torch

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    usd_member = Object(
        name="usd_member",
        object_type=ObjectType.RIGID,
        usd_path=_write_rigid_cube_usd(tmp_path / "usd_member.usda"),
        scale=(2.0, 2.0, 2.0),
    )
    usd_member.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.5)
    primitive_member = Object(
        name="primitive_member",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            size=(0.1, 0.2, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            activate_contact_sensors=True,
        ),
    )
    object_set = RigidObjectSet(name="mixed_members", objects=[usd_member, primitive_member])

    spawn_cfg = object_set.object_cfg.spawn
    assert isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg)
    assert spawn_cfg.random_choice is False
    # None keeps each member's own contact-sensor setting instead of overriding it.
    assert spawn_cfg.activate_contact_sensors is None
    usd_entry, primitive_entry = spawn_cfg.assets_cfg
    assert usd_entry.usd_path == usd_member.usd_path
    assert usd_entry.scale == (2.0, 2.0, 2.0)
    assert usd_entry.mass_props.mass == 0.5
    assert usd_entry.activate_contact_sensors is True
    assert isinstance(primitive_entry, sim_utils.CuboidCfg)
    assert primitive_entry.size == (0.1, 0.2, 0.3)
    assert primitive_entry.mass_props.mass == 1.5
    assert object_set.usd_path is None
    # Both members have their rigid body at the spawn root.
    assert object_set.get_contact_sensor_prim_path() == object_set.prim_path
    # The tallest member, the 0.3 m cuboid, gives the single bounding box.
    assert torch.allclose(object_set.get_bounding_box().size, torch.tensor([[0.1, 0.2, 0.3]]))

    # The set spawns copies: later member edits do not reach it, and edits of its entries do not reach the members.
    usd_member.object_cfg.spawn.mass_props.mass = 4.0
    assert usd_entry.mass_props.mass == 0.5
    primitive_entry.size = (1.0, 1.0, 1.0)
    assert primitive_member.object_cfg.spawn.size == (0.1, 0.2, 0.3)
    return True


def _test_single_member_object_set_spawns_the_member_spawn_cfg(simulation_app, tmp_path):
    """A one-member set spawns a plain copy of the member's spawn configuration."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    member = Object(
        name="only_member",
        object_type=ObjectType.RIGID,
        usd_path=_write_rigid_cube_usd(tmp_path / "only_member.usda"),
        scale=(1.5, 1.5, 1.5),
    )
    object_set = RigidObjectSet(name="single_member", objects=[member])
    spawn_cfg = object_set.object_cfg.spawn
    assert type(spawn_cfg) is type(member.object_cfg.spawn)
    assert spawn_cfg is not member.object_cfg.spawn
    assert spawn_cfg.usd_path == member.usd_path
    assert spawn_cfg.scale == (1.5, 1.5, 1.5)
    return True


def _test_object_set_rejects_object_set_members(simulation_app, tmp_path):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    first_member = Object(
        name="first_member", object_type=ObjectType.RIGID, usd_path=_write_rigid_cube_usd(tmp_path / "first.usda")
    )
    second_member = Object(
        name="second_member", object_type=ObjectType.RIGID, usd_path=_write_rigid_cube_usd(tmp_path / "second.usda")
    )
    # A one-member set spawns a plain config, so only the explicit check rejects it.
    inner_sets = [
        RigidObjectSet(name="one_member_set", objects=[first_member]),
        RigidObjectSet(name="two_member_set", objects=[first_member, second_member]),
    ]
    for inner_set in inner_sets:
        with pytest.raises(AssertionError, match="cannot contain another object set") as error_info:
            RigidObjectSet(name="outer_set", objects=[inner_set, second_member])
        assert "outer_set" in str(error_info.value) and inner_set.name in str(error_info.value)
    return True


def _test_object_set_addons_follow_object_rules(simulation_app):
    """A set-level asset_cfg_addon reaches the set's RigidObjectCfg; a set-level spawn_cfg_addon is rejected."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    can_a, can_b, _bbox_a, _bbox_b = _make_two_primitive_members()
    with _patch_contact_body_path():
        debug_set = RigidObjectSet(name="debug_cans", objects=[can_a, can_b], asset_cfg_addon={"debug_vis": True})
        assert debug_set.object_cfg.debug_vis is True
        with pytest.raises(AssertionError, match="spawn_cfg_addon cannot be combined with spawner_cfg"):
            RigidObjectSet(
                name="tagged_cans", objects=[can_a, can_b], spawn_cfg_addon={"semantic_tags": [("class", "can")]}
            )
    return True


def _test_object_set_rejects_set_level_scale(simulation_app):
    """A set takes each member's scale from the member, so a set-level scale is rejected."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    can_a, can_b, _bbox_a, _bbox_b = _make_two_primitive_members()
    with _patch_contact_body_path():
        with pytest.raises(AssertionError, match="Set the scale on the member objects"):
            RigidObjectSet(name="scaled_cans", objects=[can_a, can_b], scale=(2.0, 2.0, 2.0))
    return True


def _test_object_set_rejects_member_bounding_box_override(simulation_app):
    """A member with an explicit bounding_box is rejected, because set bounds come from the spawn configurations."""
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    can_a, can_b, _bbox_a, _bbox_b = _make_two_primitive_members()
    can_b.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.3))
    with pytest.raises(AssertionError, match="sets bounding_box") as error_info:
        RigidObjectSet(name="overridden_cans", objects=[can_a, can_b])
    assert "overridden_cans" in str(error_info.value) and "can_b" in str(error_info.value)
    return True


def _test_relation_free_object_set_is_a_passive_obstacle(simulation_app, tmp_path):
    """A relation-free set with a fixed pose blocks placement like other fixed rigid objects without a usd_path."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.passive_collision_objects import discover_passive_assets
    from isaaclab_arena.utils.pose import Pose

    fixed_pose = Pose(position_xyz=(0.0, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    member_usd_path = _write_rigid_cube_usd(tmp_path / "member.usda")
    members = [
        Object(name="first_member", object_type=ObjectType.RIGID, usd_path=member_usd_path),
        Object(name="second_member", object_type=ObjectType.RIGID, usd_path=member_usd_path),
    ]
    fixed_set = RigidObjectSet(name="fixed_set", objects=members, initial_pose=fixed_pose)
    fixed_cube = Object(
        name="fixed_cube", object_type=ObjectType.RIGID, usd_path=member_usd_path, initial_pose=fixed_pose
    )
    fixed_primitive = Object(
        name="fixed_primitive",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(size=(0.1, 0.2, 0.3), rigid_props=sim_utils.RigidBodyPropertiesCfg()),
        initial_pose=fixed_pose,
    )
    assert discover_passive_assets([fixed_set, fixed_cube, fixed_primitive]) == [fixed_set, fixed_cube, fixed_primitive]
    primitive_bounds = fixed_primitive.get_world_bounding_box()
    assert torch.allclose(primitive_bounds.min_point, torch.tensor([[-0.05, -0.1, -0.05]]))
    assert torch.allclose(primitive_bounds.max_point, torch.tensor([[0.05, 0.1, 0.25]]))
    return True


def _test_live_object_set_keeps_member_mass_and_scale(simulation_app, tmp_path):
    """Each environment spawns one whole member: its own mass together with its own scale."""
    import isaaclab.sim as sim_utils
    import warp as wp
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    light_member = Object(
        name="light_member",
        object_type=ObjectType.RIGID,
        usd_path=_write_rigid_cube_usd(tmp_path / "light_member.usda"),
        scale=(1.0, 1.0, 1.0),
    )
    heavy_member = Object(
        name="heavy_member",
        object_type=ObjectType.RIGID,
        usd_path=_write_rigid_cube_usd(tmp_path / "heavy_member.usda", body_path="/Object/Group/Body"),
        scale=(2.0, 2.0, 2.0),
    )
    light_member.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.5)
    heavy_member.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=2.0)
    with patch("isaaclab_arena.utils.usd.rigid_body_path_alignment.get_arena_asset_cache_dir", return_value=tmp_path):
        object_set = RigidObjectSet(
            name="weighted_set",
            objects=[light_member, heavy_member],
            random_choice=True,
            initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )
    assert object_set.get_contact_sensor_prim_path() == f"{object_set.prim_path}/rigid_body"
    # The builder keeps indices already in the spawn configuration. They differ from Isaac Lab's default
    # round robin (0, 1, 0, 1), so the check below fails unless the clone plan pins each environment's member.
    member_indices_by_env = [1, 0, 0, 1]
    object_set.object_cfg.spawn.asset_indices_by_env = tuple(member_indices_by_env)
    mass_and_scale_by_member = [(0.5, 1.0), (2.0, 2.0)]
    expected_mass_and_scale_by_env = [mass_and_scale_by_member[index] for index in member_indices_by_env]

    environment = IsaacLabArenaEnvironment(
        name="object_set_member_mass_and_scale_test", scene=Scene(assets=[object_set])
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "4"])
    env = ArenaEnvBuilder(environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        env.reset()
        masses = wp.to_torch(env.unwrapped.scene[object_set.name].data.body_mass)[:, 0].tolist()
        stage = get_current_stage()
        mass_and_scale_by_env = []
        for env_index, mass in enumerate(masses):
            root_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/{object_set.name}")
            scale = root_prim.GetAttribute("xformOp:scale").Get()
            mass_and_scale_by_env.append((round(mass, 4), round(float(scale[0]), 4)))
        assert mass_and_scale_by_env == expected_mass_and_scale_by_env, mass_and_scale_by_env
    finally:
        env.close()
    return True


def _build_and_reset_env(simulation_app, scene_assets, env_name="object_set_test", task=None):
    """Build arena env with given scene and optional task, then reset. Returns env (caller must close)."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    scene = Scene(assets=scene_assets)
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name=env_name,
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()
    return env


def _run_pick_and_place_object_set_test(
    simulation_app,
    obj_set,
    object_set_prim_path,
    path_contains,
    initial_pose=None,
):
    """Build env with one object set and PickAndPlaceTask, run common assertions, close. path_contains: str or list[str] of length NUM_ENVS."""
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    if initial_pose is not None:
        obj_set.set_initial_pose(initial_pose)
    scene_assets = [background, obj_set, destination_location]
    task = PickAndPlaceTask(
        pick_up_object=obj_set,
        destination_location=destination_location,
        background_scene=background,
    )
    env = _build_and_reset_env(
        simulation_app,
        scene_assets,
        env_name="pick_and_place_object_set_test",
        task=task,
    )
    try:
        if isinstance(path_contains, str):
            path_contains = [path_contains] * NUM_ENVS
        for i in range(NUM_ENVS):
            path = get_asset_usd_path_from_prim_path(
                prim_path=object_set_prim_path.replace(".*", str(i)),
                stage=get_current_stage(),
            )
            assert path is not None, "Path is None"
            assert path_contains[i] in path, f"Path does not contain {path_contains[i]!r}: {path}"
        if initial_pose is not None:
            assert obj_set.get_initial_pose() is not None, "Initial pose is None"
        assert env.unwrapped.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
        assert (
            env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
        ), "Contact sensor data is None"
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()


def _test_empty_object_set(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    try:
        RigidObjectSet(name="empty_object_set", objects=[])
    except Exception:
        return True
    return False


def _test_articulation_object_set(simulation_app):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    can_a, _can_b, _bbox_a, _bbox_b = _make_two_primitive_members()
    door = Object(name="door", object_type=ObjectType.ARTICULATION, usd_path="/tmp/door.usd")
    try:
        RigidObjectSet(name="articulation_object_set", objects=[can_a, door])
    except AssertionError as exc:
        return "contain only rigid objects" in str(exc)
    return False


def _test_single_object_in_one_object_set(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    obj_set = RigidObjectSet(name="single_object_set", objects=[cracker_box], prim_path=OBJECT_SET_1_PRIM_PATH)
    obj_set.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    scene = Scene(assets=[background, obj_set, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=obj_set, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="single_object_set_test",
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    try:
        for i in range(NUM_ENVS):
            # Construct the actual prim path for this environment
            path = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            assert "cracker_box.usd" in path, "Path does not contain cracker_box.usd"
            assert obj_set.get_initial_pose() is not None, "Initial pose is None"

        assert env.unwrapped.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
        assert (
            env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
        ), "Contact sensor data is None"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multi_objects_in_one_object_set(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    obj_set = RigidObjectSet(
        name="multi_object_sets", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=obj_set, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_objects_in_one_object_set_test",
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    assert env.unwrapped.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
    assert (
        env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
    ), "Contact sensor data is None"

    # replace * in OBJECT_SET_PRIM_PATH with env_index
    object_paths = []
    try:
        for i in range(NUM_ENVS):

            path = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            object_paths.append(path)
        assert len(object_paths) == NUM_ENVS, "Object_paths length is not equal to NUM_ENVS"
        # We check the file names instead of the paths because objects may be cached
        object_file_names = [os.path.basename(path) for path in object_paths]
        assert (
            os.path.basename(cracker_box.usd_path) in object_file_names
        ), "Cracker box USD path is not in Object_paths"
        assert os.path.basename(sugar_box.usd_path) in object_file_names, "Sugar box USD path is not in Object_paths"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multi_object_sets(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()

    obj_set_1 = RigidObjectSet(
        name="multi_object_sets_1", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    obj_set_2 = RigidObjectSet(
        name="multi_object_sets_2", objects=[sugar_box, mustard_bottle], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set_1, obj_set_2])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_object_sets_test",
        embodiment=embodiment,
        scene=scene,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    try:
        object_1_paths = []
        object_2_paths = []
        for i in range(NUM_ENVS):

            path_1 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            path_2 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            object_1_paths.append(path_1)
            object_2_paths.append(path_2)
            assert path_1 is not None, (
                "Path_1 from Prim Path " + OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            assert path_2 is not None, (
                "Path_2 from Prim Path " + OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
        assert len(object_1_paths) == NUM_ENVS, "Object_1_paths length is not equal to NUM_ENVS"
        assert len(object_2_paths) == NUM_ENVS, "Object_2_paths length is not equal to NUM_ENVS"
        # Check that each object in the set turns up in one of the environments
        # NOTE(alexmillane): We check the file names instead of the paths because objects may be cached
        object_1_file_names = [os.path.basename(path) for path in object_1_paths]
        object_2_file_names = [os.path.basename(path) for path in object_2_paths]
        assert (
            os.path.basename(cracker_box.usd_path) in object_1_file_names
        ), "Cracker box USD path is not in Object_1_paths"
        assert (
            os.path.basename(sugar_box.usd_path) in object_1_file_names
        ), "Sugar box USD path is not in Object_1_paths"
        assert (
            os.path.basename(sugar_box.usd_path) in object_2_file_names
        ), "Sugar box USD path is not in Object_2_paths"
        assert (
            os.path.basename(mustard_bottle.usd_path) in object_2_file_names
        ), "Mustard bottle USD path is not in Object_2_paths"
        # Each set's indices are pinned in the clone plan, so environment i spawns member i % 2 of both sets.
        set_1_member_file_names = [os.path.basename(member.usd_path) for member in (cracker_box, sugar_box)]
        set_2_member_file_names = [os.path.basename(member.usd_path) for member in (sugar_box, mustard_bottle)]
        assert object_1_file_names == [set_1_member_file_names[i % 2] for i in range(NUM_ENVS)], object_1_file_names
        assert object_2_file_names == [set_2_member_file_names[i % 2] for i in range(NUM_ENVS)], object_2_file_names
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_object_set_with_robot_mounted_cameras(simulation_app) -> bool:
    """An object set clones correctly in a scene whose cameras are mounted on the robot.

    An object set spawns one spawn entry per member, which puts the scene on Isaac Lab's
    heterogeneous clone-plan path: every cfg gets its own destination template instead of a
    single env-root one. DROID's cameras live under the robot, so their templates nest
    inside the robot's, and resolving them used to raise. Needs more than one env; a single
    env takes the homogeneous fast path and never builds the nested templates.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras"])
    args_cli.num_envs = NUM_ENVS_WITH_CAMERAS

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    sweet_potato = asset_registry.get_asset_by_name("sweet_potato")()
    jug = asset_registry.get_asset_by_name("jug")()

    object_set = RigidObjectSet(name="object_set", objects=[sweet_potato, jug])
    object_set.set_initial_pose(
        Pose(position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5), rotation_xyzw=(0, 0, 0, 1))
    )

    scene = Scene(assets=[background, object_set])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_set_with_cameras_test",
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True),
        scene=scene,
    )

    # Scene construction is what used to raise, so reaching the first observation is the
    # regression signal.
    builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered()
    env.reset()

    # The observation dict is the env's own buffer, so it must be read before env.close().
    try:
        for _ in tqdm.tqdm(range(NUM_STEPS_WITH_CAMERAS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, _, _, _, _ = env.step(actions)

                assert "camera_obs" in obs, f"No camera observation group; got groups {sorted(obs)}"
                camera_obs = obs["camera_obs"]
                # Every camera nested under the robot must survive cloning: the two external
                # ones under panda_link0 and the wrist one under the gripper.
                for camera_key in ("external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb"):
                    assert camera_key in camera_obs, f"Missing {camera_key!r}; got {sorted(camera_obs)}"
                    images = camera_obs[camera_key]
                    num_envs = images.shape[0]
                    assert num_envs == NUM_ENVS_WITH_CAMERAS, f"{camera_key} has {num_envs} envs"
                    assert images.shape[3] == 3, f"{camera_key} rgb observation does not have three channels"
                    assert images.any(), f"{camera_key} observation contains only 0s"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def test_empty_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_empty_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_empty_object_set.__name__} failed"


def test_object_set_assignment_drives_bounds_and_contact_paths():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_assignment_drives_bounds_and_contact_paths,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_assignment_drives_bounds_and_contact_paths.__name__} failed"


def test_two_object_sets_resolve_normalized_usd_contact_paths(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_two_object_sets_resolve_normalized_usd_contact_paths,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result


def test_articulation_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_articulation_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_articulation_object_set.__name__} failed"


def test_single_object_in_one_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_single_object_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_single_object_in_one_object_set.__name__} failed"


def test_multi_objects_in_one_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_multi_objects_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_objects_in_one_object_set.__name__} failed"


def test_multi_object_sets():
    result = run_function_with_persistent_simulation_app(
        _test_multi_object_sets,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_object_sets.__name__} failed"


@pytest.mark.with_cameras
def test_object_set_with_robot_mounted_cameras():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_with_robot_mounted_cameras,
        headless=HEADLESS,
        enable_cameras=True,
    )
    assert result, f"Test {_test_object_set_with_robot_mounted_cameras.__name__} failed"


def test_object_set_spawns_members_from_their_own_spawn_cfgs(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_object_set_spawns_members_from_their_own_spawn_cfgs,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_object_set_spawns_members_from_their_own_spawn_cfgs.__name__} failed"


def test_single_member_object_set_spawns_the_member_spawn_cfg(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_single_member_object_set_spawns_the_member_spawn_cfg,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_single_member_object_set_spawns_the_member_spawn_cfg.__name__} failed"


def test_object_set_rejects_object_set_members(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_object_set_rejects_object_set_members,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_object_set_rejects_object_set_members.__name__} failed"


def test_object_set_addons_follow_object_rules():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_addons_follow_object_rules,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_addons_follow_object_rules.__name__} failed"


def test_object_set_rejects_set_level_scale():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_rejects_set_level_scale,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_rejects_set_level_scale.__name__} failed"


def test_object_set_rejects_member_bounding_box_override():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_rejects_member_bounding_box_override,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_rejects_member_bounding_box_override.__name__} failed"


def test_relation_free_object_set_is_a_passive_obstacle(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_relation_free_object_set_is_a_passive_obstacle,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_relation_free_object_set_is_a_passive_obstacle.__name__} failed"


def test_live_object_set_keeps_member_mass_and_scale(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_live_object_set_keeps_member_mass_and_scale,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_live_object_set_keeps_member_mass_and_scale.__name__} failed"


if __name__ == "__main__":
    test_empty_object_set()
    test_articulation_object_set()
    test_single_object_in_one_object_set()
    test_multi_objects_in_one_object_set()
    test_multi_object_sets()
    test_object_set_with_robot_mounted_cameras()
