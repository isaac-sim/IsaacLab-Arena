# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for scenes whose objects have several spawn entries, built through ArenaEnvBuilder."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01


def _make_table_anchor():
    """Return a fixed 2 m by 2 m table anchor, 0.1 m thick, whose top surface is at TABLE_TOP_Z."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    return Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(2.0, 2.0, 0.1)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )


# Compose-level builder tests. Each compose builds fresh objects, because the indices stay in object_cfg.spawn.

ORDERED_SET_HEIGHTS = (0.1, 0.3)
RANDOM_SET_HEIGHTS = (0.2, 0.4, 0.6)
RELATION_FREE_SET_HEIGHTS = (0.2, 0.4, 0.6)
PLACEMENT_SEED = 11


def _make_cuboid_set(name, heights, random_choice):
    """Return a RigidObjectSet with one 0.1 m by 0.1 m rigid cuboid member per height."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    members = []
    for member_index, height in enumerate(heights):
        member_spawn_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, height), rigid_props=sim_utils.RigidBodyPropertiesCfg())
        member_name = f"{name}_member_{member_index}"
        members.append(Object(name=member_name, object_type=ObjectType.RIGID, spawner_cfg=member_spawn_cfg))
    return RigidObjectSet(name=name, objects=members, random_choice=random_choice)


def _make_placed_environment(placer_params=None):
    """Return an environment with a table anchor, an ordered set and a random set, both On(table), and the sets."""
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene

    table = _make_table_anchor()
    ordered_set = _make_cuboid_set("ordered_set", ORDERED_SET_HEIGHTS, random_choice=False)
    random_set = _make_cuboid_set("random_set", RANDOM_SET_HEIGHTS, random_choice=True)
    ordered_set.add_relation(On(table))
    random_set.add_relation(On(table))
    environment = IsaacLabArenaEnvironment(
        name="builder_assignment_test",
        scene=Scene(assets=[table, ordered_set, random_set]),
        placer_params=placer_params,
    )
    return environment, ordered_set, random_set


def _compose_placed_environment(seed_source):
    """Compose the placed environment for 4 envs with PLACEMENT_SEED set on seed_source; return the cfg and sets."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

    placer_params = ObjectPlacerParams(placement_seed=PLACEMENT_SEED) if seed_source == "placer_params" else None
    builder_placement_seed = PLACEMENT_SEED if seed_source == "builder_cfg" else None
    environment, ordered_set, random_set = _make_placed_environment(placer_params)
    # resolve_on_reset=False keeps one construction layout per environment; True gives every env env 0's layout.
    builder_cfg = ArenaEnvBuilderCfg(num_envs=4, placement_seed=builder_placement_seed, resolve_on_reset=False)
    env_cfg, _ = ArenaEnvBuilder(environment, builder_cfg).compose_manager_cfg()
    return env_cfg, ordered_set, random_set


def _get_solved_z_by_env(obj):
    """Return the construction z of obj in each environment."""
    return [pose.position_xyz[2] for pose in obj.get_initial_pose().poses]


def _test_builder_assigns_relation_placed_sets_before_placement(simulation_app, seed_source):
    """The builder assigns both placed sets before placement, and each env's z follows its assigned member."""
    import torch

    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg

    env_cfg, ordered_set, random_set = _compose_placed_environment(seed_source)

    for obj in (ordered_set, random_set):
        # The scene configuration holds its own copy of each set's spawn configuration, indices included.
        scene_spawn_cfg = getattr(env_cfg.scene, obj.name).spawn
        assert isinstance(scene_spawn_cfg, PinnedMultiAssetSpawnerCfg), obj.name
        assert scene_spawn_cfg is not obj.object_cfg.spawn, obj.name
        assert scene_spawn_cfg.asset_indices_by_env == obj.object_cfg.spawn.asset_indices_by_env, obj.name
    assert ordered_set.object_cfg.spawn.asset_indices_by_env == (0, 1, 0, 1)
    if seed_source != "none":
        # The ordered set is assigned first (offset 0), so the random set draws with the seed plus 1.
        generator = torch.Generator().manual_seed(PLACEMENT_SEED + 1)
        expected_random_indices = tuple(torch.randint(0, 3, (4,), generator=generator).tolist())
        assert random_set.object_cfg.spawn.asset_indices_by_env == expected_random_indices
    for obj, heights in ((ordered_set, ORDERED_SET_HEIGHTS), (random_set, RANDOM_SET_HEIGHTS)):
        asset_indices_by_env = obj.object_cfg.spawn.asset_indices_by_env
        expected_z_by_env = [TABLE_TOP_Z + ON_CLEARANCE_M + heights[index] / 2 for index in asset_indices_by_env]
        assert _get_solved_z_by_env(obj) == pytest.approx(expected_z_by_env, abs=1e-3), obj.name

    if seed_source != "none":
        # A fresh build with the same seed reproduces the indices and the solved heights.
        _, repeated_ordered_set, repeated_random_set = _compose_placed_environment(seed_source)
        for obj, repeated_obj in ((ordered_set, repeated_ordered_set), (random_set, repeated_random_set)):
            assert repeated_obj.object_cfg.spawn.asset_indices_by_env == obj.object_cfg.spawn.asset_indices_by_env
        assert _get_solved_z_by_env(repeated_random_set) == pytest.approx(_get_solved_z_by_env(random_set), abs=1e-6)
    return True


def _make_relation_free_set(name, random_choice):
    """Return a three-member cuboid set at a fixed pose, with no relations."""
    from isaaclab_arena.utils.pose import Pose

    object_set = _make_cuboid_set(name, RELATION_FREE_SET_HEIGHTS, random_choice=random_choice)
    object_set.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    return object_set


def _test_builder_assigns_sets_without_relation_solving(simulation_app):
    """A relation-free random set is assigned and seeded by the placement seed, also with relation solving off."""
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    random_set = _make_relation_free_set("random_set", random_choice=True)
    environment = IsaacLabArenaEnvironment(name="relation_free_assignment_test", scene=Scene(assets=[random_set]))
    builder_cfg = ArenaEnvBuilderCfg(num_envs=4, solve_relations=False, placement_seed=5)
    env_cfg, _ = ArenaEnvBuilder(environment, builder_cfg).compose_manager_cfg()

    # The only multi-asset object gets offset 0. The draw (2, 0, 1, 0) differs from the round robin (0, 1, 2, 0).
    expected_indices = tuple(torch.randint(0, 3, (4,), generator=torch.Generator().manual_seed(5)).tolist())
    assert random_set.object_cfg.spawn.asset_indices_by_env == expected_indices
    # The scene configuration holds its own copy with the same indices.
    scene_spawn_cfg = env_cfg.scene.random_set.spawn
    assert scene_spawn_cfg is not random_set.object_cfg.spawn
    assert scene_spawn_cfg.asset_indices_by_env == expected_indices
    return True


def _test_environment_rejects_an_unpinned_multi_entry_spawner(simulation_app):
    """The environment rejects a multi-entry spawner that is not pinned when it builds its scene."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    def replace_pinned_spawner(env_cfg):
        # A YAML env_cfg_override with _target_: isaaclab.sim.MultiAssetSpawnerCfg has the same effect.
        pinned_spawn_cfg = env_cfg.scene.ordered_set.spawn
        env_cfg.scene.ordered_set.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=pinned_spawn_cfg.assets_cfg)
        return env_cfg

    environment = IsaacLabArenaEnvironment(
        name="unpinned_spawner_env_build_test",
        scene=Scene(assets=[_make_relation_free_set("ordered_set", random_choice=False)]),
        env_cfg_callback=replace_pinned_spawner,
    )
    builder = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=4, solve_relations=False))
    with pytest.raises(AssertionError, match="'ordered_set' spawns several entries from a MultiAssetSpawnerCfg"):
        env = builder.make_registered()
        # Reached only when nothing checks the scene; close the environment so later tests can build theirs.
        env.close()
    return True


@pytest.mark.parametrize("seed_source", ["builder_cfg", "placer_params", "none"])
def test_builder_assigns_relation_placed_sets_before_placement(seed_source):
    assert run_function_with_persistent_simulation_app(
        _test_builder_assigns_relation_placed_sets_before_placement, seed_source=seed_source
    )


def test_builder_assigns_sets_without_relation_solving():
    assert run_function_with_persistent_simulation_app(_test_builder_assigns_sets_without_relation_solving)


def test_environment_rejects_an_unpinned_multi_entry_spawner():
    assert run_function_with_persistent_simulation_app(
        _test_environment_rejects_an_unpinned_multi_entry_spawner, headless=HEADLESS
    )


# Live scene: each environment spawns the spawn entry its indices name, with that entry's geometry and physics.

# Edge lengths of the collision cube in both fixture assets, along the rigid body's own axes.
CUBE_SIZE_IN_BODY_FRAME = (0.12, 0.24, 0.36)
ROOT_BODY_MASS = 0.2
NESTED_BODY_MASS = 0.5
# In the nested asset the body sits below a translated group and is itself translated and turned about z.
NESTED_GROUP_TRANSLATION = (0.4, -0.2, 0.3)
NESTED_BODY_TRANSLATION = (0.1, 0.2, 0.3)


def _write_rigid_asset(path, nested: bool, mass: float) -> None:
    """Write a USD asset with one gravity-free rigid body that holds a collision cube.

    Args:
        path: Output file path.
        nested: Whether the body is a child of a group below the default prim, translated and turned
            90 degrees about z, instead of the default prim itself.
        mass: Mass authored on the body.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    body = root
    if nested:
        group = UsdGeom.Xform.Define(stage, "/Asset/Group")
        group.AddTranslateOp().Set(Gf.Vec3d(*NESTED_GROUP_TRANSLATION))
        body = UsdGeom.Xform.Define(stage, "/Asset/Group/Body").GetPrim()
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(*NESTED_BODY_TRANSLATION))
        UsdGeom.Xformable(body).AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(body)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(True)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
    cube = UsdGeom.Cube.Define(stage, str(body.GetPath()) + "/Cube")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*CUBE_SIZE_IN_BODY_FRAME))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()


def _get_expected_values_per_env(spawn_entries, asset_indices_by_env):
    """Return the mass, cube size and bounds that each environment's assigned spawn entry should give.

    Args:
        spawn_entries: (spawn scale, nested) of each spawn entry, in entry order.
        asset_indices_by_env: Spawn entry index of each environment.

    Returns:
        Body masses (num_envs,), cube sizes in the body frame (num_envs, 3), and bounds sizes and
        centres in the default prim's frame (num_envs, 3).
    """
    import torch

    masses = []
    body_frame_sizes = []
    bounds_sizes = []
    bounds_centers = []
    for asset_index in asset_indices_by_env:
        scale, nested = spawn_entries[asset_index]
        body_frame_size = [scale * edge for edge in CUBE_SIZE_IN_BODY_FRAME]
        body_frame_sizes.append(body_frame_size)
        if nested:
            masses.append(NESTED_BODY_MASS)
            # The body is turned 90 degrees about z, which swaps its x and y extents in the default prim's frame.
            bounds_sizes.append([body_frame_size[1], body_frame_size[0], body_frame_size[2]])
            body_position = [
                scale * (group + body) for group, body in zip(NESTED_GROUP_TRANSLATION, NESTED_BODY_TRANSLATION)
            ]
            bounds_centers.append(body_position)
        else:
            masses.append(ROOT_BODY_MASS)
            bounds_sizes.append(body_frame_size)
            bounds_centers.append([0.0, 0.0, 0.0])
    return (
        torch.tensor(masses),
        torch.tensor(body_frame_sizes),
        torch.tensor(bounds_sizes),
        torch.tensor(bounds_centers),
    )


def _test_environment_spawns_the_assigned_spawn_entry_in_every_environment(simulation_app, tmp_path):
    """Each environment spawns its assigned entry of every object, with that entry's bounds, mass and inertia."""
    import torch
    from unittest.mock import patch

    import warp as wp
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd.rigid_body_path_alignment import align_rigid_body_paths

    num_envs = 9
    root_asset_path = tmp_path / "root.usda"
    nested_asset_path = tmp_path / "nested.usda"
    _write_rigid_asset(root_asset_path, nested=False, mass=ROOT_BODY_MASS)
    _write_rigid_asset(nested_asset_path, nested=True, mass=NESTED_BODY_MASS)
    # (spawn scale, nested) of each spawn entry. Both objects have three entries, so a swap of their clone-plan
    # columns keeps every entry count valid and only the per-environment checks below catch it.
    spawn_entries_by_object = {
        "pickup": [(1.0, False), (1.5, True), (0.75, False)],
        "destination": [(0.5, True), (1.0, True), (2.0, True)],
    }
    initial_positions_by_object = {"pickup": (0.0, 0.0, 2.0), "destination": (1.0, 0.0, 2.0)}

    def make_multi_asset_object(name):
        spawn_cfgs = []
        for scale, nested in spawn_entries_by_object[name]:
            usd_path = nested_asset_path if nested else root_asset_path
            spawn_cfgs.append(UsdFileCfg(usd_path=str(usd_path), scale=(scale,) * 3, activate_contact_sensors=True))
        spawner_cfg = PinnedMultiAssetSpawnerCfg(
            assets_cfg=align_rigid_body_paths(spawn_cfgs),
            # Arena assigns the entries. Isaac Lab ignores this flag and warns when it is True.
            random_choice=False,
            # A value here would replace every entry's own setting.
            activate_contact_sensors=None,
        )
        return Object(
            name=name,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawner_cfg,
            initial_pose=Pose(position_xyz=initial_positions_by_object[name], rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )

    # The pickup mixes body depths, so align_rigid_body_paths writes copies with the body at /rigid_body.
    with patch("isaaclab_arena.utils.usd.rigid_body_path_alignment.get_arena_asset_cache_dir", return_value=tmp_path):
        pickup = make_multi_asset_object("pickup")
        destination = make_multi_asset_object("destination")
    # Indices already in a spawn configuration are kept. These differ from the round robin and from the order
    # Isaac Lab's default sequential strategy gives the first of two three-entry objects, (0, 0, 0, 1, 1, 1, 2, 2, 2).
    pickup.object_cfg.spawn.asset_indices_by_env = (2, 0, 1, 1, 2, 0, 0, 1, 2)
    expected_indices_by_object = {
        "pickup": (2, 0, 1, 1, 2, 0, 0, 1, 2),
        "destination": (0, 1, 2, 0, 1, 2, 0, 1, 2),
    }

    def configure_contacts(env_cfg):
        env_cfg.scene.pickup_contacts = pickup.get_contact_sensor_cfg(destination)
        return env_cfg

    arena_environment = IsaacLabArenaEnvironment(
        name="object_asset_assignment_scene_test",
        scene=Scene(assets=[pickup, destination]),
        env_cfg_callback=configure_contacts,
    )
    builder = ArenaEnvBuilder(
        arena_environment, ArenaEnvBuilderCfg(num_envs=num_envs, env_spacing=4.0, seed=19, disable_fabric=True)
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    # The builder wrote round-robin indices for the destination and kept the pickup's.
    for obj in (pickup, destination):
        assert obj.object_cfg.spawn.asset_indices_by_env == expected_indices_by_object[obj.name], obj.name
    recorded_clone_strategy = env_cfg.scene.clone_cfg.clone_strategy

    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        # parse_env_cfg hands the registered cfg instance itself to the env, and the strategy that pins the
        # entries is installed only while the scene is built.
        assert env.unwrapped.cfg is env_cfg
        assert env_cfg.scene.clone_cfg.clone_strategy is recorded_clone_strategy
        env.reset()
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        runtime_scene = env.unwrapped.scene
        # The scene keeps the scene configuration the strategy read the indices from.
        assert runtime_scene.cfg is env_cfg.scene
        clone_plan = runtime_scene.clone_plan
        for obj in (pickup, destination):
            expected_indices = expected_indices_by_object[obj.name]
            # The clone plan has one row per spawn entry of this object; column j is environment j.
            object_rows = list(clone_plan.cfg_rows[id(getattr(runtime_scene.cfg, obj.name))])
            entry_mask = clone_plan.clone_mask[object_rows]
            assert torch.equal(entry_mask.sum(dim=0).cpu(), torch.ones(num_envs, dtype=torch.long))
            spawned_indices = tuple(entry_mask.long().argmax(dim=0).tolist())
            assert (
                spawned_indices == expected_indices
            ), f"{obj.name}: clone plan spawns {spawned_indices}, assignment is {expected_indices}"

            runtime_object = runtime_scene[obj.name]
            assert runtime_object.num_instances == num_envs
            assert runtime_object.num_bodies == 1
            expected_masses, body_frame_sizes, expected_bounds_sizes, expected_bounds_centers = (
                _get_expected_values_per_env(spawn_entries_by_object[obj.name], expected_indices)
            )
            masses = wp.to_torch(runtime_object.data.body_mass).cpu().flatten()
            torch.testing.assert_close(masses, expected_masses)
            bounds = obj.get_bounding_box_per_env(num_envs)
            torch.testing.assert_close(bounds.size, expected_bounds_sizes, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(bounds.center, expected_bounds_centers, atol=1e-5, rtol=1e-5)
            # PhysX computes the inertia of the cube in the body frame, where the cube is not turned.
            squared_sizes = body_frame_sizes.square()
            sums_of_other_squared_sizes = squared_sizes.sum(dim=1, keepdim=True) - squared_sizes
            expected_diagonals = expected_masses[:, None] * sums_of_other_squared_sizes / 12
            inertias = wp.to_torch(runtime_object.data.body_inertia).cpu().reshape(num_envs, 3, 3)
            torch.testing.assert_close(inertias, torch.diag_embed(expected_diagonals), atol=1e-6, rtol=1e-4)
        force_matrix = runtime_scene.sensors["pickup_contacts"].data.force_matrix_w
        assert force_matrix is not None
        assert wp.to_torch(force_matrix).shape == (num_envs, 1, 1, 3)

        env.reset()
        for obj in (pickup, destination):
            root_positions = wp.to_torch(runtime_scene[obj.name].data.root_pos_w) - runtime_scene.env_origins
            expected_position = torch.tensor(obj.initial_pose.position_xyz, device=root_positions.device)
            torch.testing.assert_close(root_positions, expected_position.expand(num_envs, 3), atol=1e-5, rtol=1e-5)
            root_orientations = wp.to_torch(runtime_scene[obj.name].data.root_quat_w)
            identity_xyzw = torch.tensor((0.0, 0.0, 0.0, 1.0), device=root_orientations.device)
            torch.testing.assert_close(root_orientations, identity_xyzw.expand(num_envs, 4), atol=1e-5, rtol=1e-5)
    finally:
        env.close()
    return True


def test_environment_spawns_the_assigned_spawn_entry_in_every_environment(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_environment_spawns_the_assigned_spawn_entry_in_every_environment,
        headless=HEADLESS,
        tmp_path=tmp_path,
    )
    assert result, f"Test {_test_environment_spawns_the_assigned_spawn_entry_in_every_environment.__name__} failed"
