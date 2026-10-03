# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Saved-layout replay and reset ownership."""


from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "test_data/placement_replay.yaml"
LAYOUTS = SOURCE.with_suffix(".jsonl")


def _test_companion_cache_round_trip(simulation_app, tmp_path):
    import torch
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
    from isaaclab_arena.relations.relation_solver import RelationSolver

    cache = PlacementLayouts.from_episode_jsonl(LAYOUTS)
    data = yaml.safe_load(SOURCE.read_text())
    data["objects"][3]["params"] = {"initial_pose": cache.poses["cube_3"][0].to_dict()}
    data["relations"] = [relation for relation in data["relations"] if relation["subject"] != "cube_3"]
    data["relations"].append({"kind": "is_anchor", "subject": "cube_3"})
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    spec = ArenaEnvGraphSpec.from_yaml(source)
    with (
        patch.object(RelationSolver, "solve", side_effect=AssertionError("Cached replay must not solve")),
        patch.object(
            PlacementValidationRunner,
            "validate_candidates",
            side_effect=AssertionError("Cached replay must not revalidate"),
        ),
    ):
        arena_env = spec.to_arena_env()
        assert arena_env.scene.assets["cube_3"].has_pose_reset_event()
        arena_env.embodiment.set_initial_pose(arena_env.embodiment.get_initial_pose())
        cfg = ArenaEnvBuilderCfg(num_envs=3, placement_layouts_path=str(LAYOUTS))
        env = ArenaEnvBuilder(arena_env, cfg).make_registered()
        try:
            scene = env.unwrapped.scene
            world = env.unwrapped.arena_world
            device = env.unwrapped.device
            cached_assets = {
                asset.get_scene_key(): asset
                for asset in arena_env.scene.assets.values()
                if asset.get_scene_key() in cache.poses
            }
            assert cached_assets.keys() == cache.poses.keys()
            assert all(not asset.has_pose_reset_event() for asset in cached_assets.values())
            robot = scene.articulations["robot"]
            robot_pose = robot.data.root_pose_w.torch.clone()
            num_resets = 4
            for iteration in range(num_resets):
                moved = robot_pose.clone()
                moved[:, 0] += 0.25
                robot.write_root_pose_to_sim(moved)
                for body in scene.rigid_objects.values():
                    displaced = body.data.root_pose_w.torch.clone()
                    displaced[:, 2] += 1.0
                    body.write_root_pose_to_sim(displaced)
                    body.write_root_velocity_to_sim(torch.ones_like(body.data.root_vel_w.torch))
                env.reset()
                torch.testing.assert_close(robot.data.root_pose_w.torch, robot_pose, atol=2e-5, rtol=0)
                for name, poses in cache.poses.items():
                    expected = torch.stack(
                        [poses[(iteration * 3 + i) % cache.num_layouts].to_tensor(device) for i in range(3)]
                    )
                    torch.testing.assert_close(world.get_pose_e(name), expected, atol=2e-5, rtol=0)
            before = {name: world.get_pose_e(name) for name in cache.poses}
            next_layout = num_resets * env.unwrapped.num_envs % cache.num_layouts
            env_ids = torch.tensor([1], device=device)
            for name in cache.poses:
                displaced = scene[name].data.root_pose_w.torch[env_ids].clone()
                displaced[:, 2] += 1.0
                scene[name].write_root_pose_to_sim(displaced, env_ids=env_ids)
                scene[name].write_root_velocity_to_sim(torch.ones((1, 6), device=device), env_ids=env_ids)
            env.unwrapped._reset_idx(env_ids)
            for name, poses in cache.poses.items():
                actual = world.get_pose_e(name)
                torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
                torch.testing.assert_close(actual[1], poses[next_layout].to_tensor(device), atol=2e-5, rtol=0)
                torch.testing.assert_close(
                    scene[name].data.root_vel_w.torch[env_ids],
                    torch.zeros((1, 6), device=device),
                    atol=0,
                    rtol=0,
                )
            env.unwrapped._reset_idx(torch.tensor([2], device=device))
            for name, poses in cache.poses.items():
                torch.testing.assert_close(
                    world.get_pose_e(name)[2],
                    poses[(next_layout + 1) % cache.num_layouts].to_tensor(device),
                    atol=2e-5,
                    rtol=0,
                )
            before = {name: world.get_pose_e(name) for name in cache.poses}
            for _ in range(200):
                scene.write_data_to_sim()
                env.unwrapped.sim.step(render=False)
                scene.update(env.unwrapped.sim.get_physics_dt())
            for name in cache.poses:
                assert float((world.get_pose_e(name)[:, :3] - before[name][:, :3]).norm(dim=-1).max()) < 0.02
        finally:
            env.close()
    return True


def test_companion_cache_round_trip(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_companion_cache_round_trip, tmp_path=tmp_path)


def _make_cached_env():
    from isaaclab_arena.assets.background_library import OfficeTableBackground
    from isaaclab_arena.assets.object_library import DexCube, DomeLight
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    table = OfficeTableBackground()
    table.set_initial_pose(Pose.identity())
    table.add_relation(IsAnchor())
    cubes = [DexCube(instance_name=f"cube_{i}") for i in range(4)]
    for cube in cubes:
        cube.add_relation(On(table))
    return IsaacLabArenaEnvironment(
        name="python_placement_replay",
        scene=Scene(assets=[table, *cubes, DomeLight()]),
        embodiment=FrankaIKEmbodiment(initial_pose=Pose((-1.0, 0.0, 0.0))),
        placer_params=ObjectPlacerParams(),
        placement_layouts=PlacementLayouts({cube.get_scene_key(): [Pose((0, 0, 1))] for cube in cubes}),
    )


def _test_cache_rejects_conflicting_configuration(simulation_app, conflict, expected_error):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.relations import RandomAroundSolution, RotateAroundSolution
    from isaaclab_arena.utils.pose import PoseRange
    from isaaclab_arena.utils.velocity import Velocity

    arena_env = _make_cached_env()
    cfg = ArenaEnvBuilderCfg()
    cube = arena_env.scene.assets["cube_0"]
    if conflict == "random-reset":
        cube.add_relation(RandomAroundSolution())
    elif conflict == "pose-range":
        cube.set_initial_pose(PoseRange(position_xyz_max=(1.0, 1.0, 1.0)))
    elif conflict == "disabled-reset":
        cube.disable_reset_pose()
    elif conflict == "initial-velocity":
        cube.set_initial_velocity(Velocity(linear_xyz=(1.0, 0.0, 0.0)))
    elif conflict == "missing-robot":
        arena_env.embodiment.add_relation(RotateAroundSolution(yaw_rad=0.5))
    elif conflict == "placement-seed":
        cfg.placement_seed = 42
    elif conflict == "placement-seed-default":
        arena_env.placer_params.placement_seed = 42
    elif conflict == "fixed-layout":
        cfg.resolve_on_reset = False
    elif conflict == "two-layout-sources":
        cfg.placement_layouts_path = "unused.jsonl"
    elif conflict == "fixed-layout-default":
        arena_env.placer_params.resolve_on_reset = False
    builder = ArenaEnvBuilder(arena_env, cfg)
    with pytest.raises(AssertionError, match=expected_error):
        builder.compose_manager_cfg()
    return True


@pytest.mark.parametrize(
    "conflict, expected_error",
    [
        ("random-reset", "cannot randomize"),
        ("pose-range", "non-fixed pose-reset"),
        ("disabled-reset", "pose resets disabled"),
        ("initial-velocity", "nonzero initial velocity"),
        ("missing-robot", "missing placed objects.*robot"),
        ("placement-seed", "placement_seed applies to solving"),
        ("placement-seed-default", "placement_seed applies to solving"),
        ("fixed-layout", "requires resolve_on_reset=True"),
        ("fixed-layout-default", "requires resolve_on_reset=True"),
        ("two-layout-sources", "not both"),
    ],
)
def test_cache_rejects_conflicting_configuration(conflict, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_cache_rejects_conflicting_configuration, conflict=conflict, expected_error=expected_error
    )


def _test_python_integer_layouts_reset_objects_and_robot(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_cached_env()
    poses = {f"cube_{i}": [Pose((i, 0, 2), (0, 0, 0, 1)), Pose((i, 1, 2), (0, 0, 0, 1))] for i in range(4)}
    poses["robot"] = [Pose((-1, 0, 0), (0, 0, 0, 1)), Pose((-1, 1, 0), (0, 0, 0, 1))]
    arena_env.placement_layouts = None
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, resolve_on_reset=True))
    arena_env.placement_layouts = PlacementLayouts(poses)
    arena_env.embodiment.set_initial_pose(arena_env.embodiment.get_initial_pose())
    assert arena_env.embodiment.has_pose_reset_event()
    arena_env.placer_params.resolve_on_reset = False
    env = builder.make_registered()
    try:
        assert not arena_env.embodiment.has_pose_reset_event()
        # Equal counts consume and wrap the whole queue on every full reset.
        for _ in range(3):
            for name in poses:
                body = env.unwrapped.scene[name]
                displaced = body.data.root_pose_w.torch.clone()
                displaced[:, 0] += 0.25
                displaced[:, 3:] = torch.tensor([0, 0, 1, 0], device=env.unwrapped.device)
                body.write_root_pose_to_sim(displaced)
                body.write_root_velocity_to_sim(torch.ones_like(body.data.root_vel_w.torch))
            env.reset()
            for name, layouts in poses.items():
                expected = torch.tensor(
                    [layouts[i % 2].position_xyz + layouts[i % 2].rotation_xyzw for i in range(2)],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)
                velocity = env.unwrapped.scene[name].data.root_vel_w.torch
                torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=0, rtol=0)
    finally:
        env.close()
    return True


def test_python_integer_layouts_reset_objects_and_robot():
    assert run_function_with_persistent_simulation_app(_test_python_integer_layouts_reset_objects_and_robot)


def _test_cli_loads_layouts_for_yaml_and_python_environments(simulation_app, tmp_path, environment):
    import sys
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    path = tmp_path / "poses.jsonl"
    if environment == "yaml":
        data = yaml.safe_load(SOURCE.read_text())
        data["embodiment"]["id"] = "arm"
        source = tmp_path / "scene.yaml"
        source.write_text(yaml.safe_dump(data))
        env_args = ["--env_spec", str(source)]
    else:
        env_args = ["droid_table_multi_object_placement"]
    with patch.object(sys, "argv", ["environment_runner.py", "--placement_layouts", str(path), *env_args]):
        args = get_isaaclab_arena_environments_cli_parser().parse_args()
    builder = get_arena_builder_from_cli(args)
    poses = {
        asset.get_scene_key(): [Pose((0.1 * i, 0, 1))]
        for i, asset in enumerate(builder.arena_env.scene.assets.values())
        if asset.get_spatial_relations() and not asset.is_anchor
    }
    poses["robot"] = [Pose((-1, 0, 0))]
    PlacementLayouts(poses).write_episode_jsonl(path, source="example")
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Cached replay must not solve")):
        cfg, _ = builder.compose_manager_cfg()
    recorded = cfg.events.cached_placement_reset.params["poses"]
    assert set(recorded) == set(poses)
    assert recorded["robot"] == [list(poses["robot"][0].position_xyz + poses["robot"][0].rotation_xyzw)]
    assert cfg.scene.robot.init_state.pos == poses["robot"][0].position_xyz
    if environment == "yaml":
        assert "arm" not in recorded
        assert cfg.scene.cube_1.init_state.pos == poses["cube_1"][0].position_xyz
    else:
        for invalid, message in (
            ({"unknown": [Pose()]}, "Unknown cached"),
            ({"robot": [Pose()]}, "missing placed"),
        ):
            invalid_path = tmp_path / f"{message.split()[0]}.jsonl"
            PlacementLayouts(invalid).write_episode_jsonl(invalid_path, source="example")
            arena_env = _make_cached_env()
            arena_env.placement_layouts = None
            builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg())
            builder.cfg.placement_layouts_path = str(invalid_path)
            with pytest.raises(AssertionError, match=message):
                builder.compose_manager_cfg()
    return True


@pytest.mark.parametrize("environment", ["yaml", "python"])
def test_cli_loads_layouts_for_yaml_and_python_environments(tmp_path, environment):
    assert run_function_with_persistent_simulation_app(
        _test_cli_loads_layouts_for_yaml_and_python_environments, tmp_path=tmp_path, environment=environment
    )


def _make_two_root_embodiment(tmp_path):
    """Build two independent procedural articulations for root replay coverage."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.embodiments.no_embodiment import EmptyActionsCfg
    from isaaclab_arena.utils.configclass import make_configclass

    path = tmp_path / "two_link_robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, x in (("base", 0.0), ("link", 0.2)):
        body = UsdGeom.Cube.Define(stage, f"/Robot/{name}")
        body.CreateSizeAttr(0.1)
        body.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.2))
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/world_joint")
    fixed.CreateBody1Rel().SetTargets(["/Robot/base"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint")
    joint.CreateBody0Rel().SetTargets(["/Robot/base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/link"])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.1, 0.0, 0.0))
    stage.GetRootLayer().Save()

    class TwoRootEmbodiment(EmbodimentBase):
        name = "two_root_test"

    embodiment = TwoRootEmbodiment()
    embodiment.action_config = EmptyActionsCfg()
    roots = []
    for name, y in (("left_robot", 0.5), ("right_robot", -0.5)):
        cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, y, 0.0), joint_pos={"joint": 0.0}),
            actuators={"joint": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
        )
        roots.append((name, ArticulationCfg, cfg))
    embodiment.scene_config = make_configclass("TwoRootSceneCfg", roots)()
    return embodiment


def _test_bimanual_root_recording_and_replay(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.recording import validate_recording_assets
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    embodiment = _make_two_root_embodiment(tmp_path)
    keys = ("left_robot", "right_robot")
    assert embodiment.get_scene_root_keys() == keys
    assert get_scene_root_owners([embodiment]) == {key: embodiment for key in keys}
    with pytest.raises(AssertionError, match="multiple asset owners"):
        get_scene_root_owners([embodiment, embodiment])
    with pytest.raises(AssertionError, match="right_robot"):
        PlacementLayouts({"left_robot": [Pose()]}).validate_assets([embodiment])

    poses = {
        "left_robot": [Pose((0.0, 0.5, 0.0)), Pose((0.2, 0.7, 0.0)), Pose((-0.2, 0.6, 0.0))],
        "right_robot": [Pose((0.1, -0.5, 0.0)), Pose((-0.1, -0.7, 0.0)), Pose((0.3, -0.6, 0.0))],
    }
    path = tmp_path / "bimanual.jsonl"
    PlacementLayouts(poses).write_episode_jsonl(path, source="settled")
    arena = IsaacLabArenaEnvironment(name="bimanual_root_replay", scene=Scene(assets=[]), embodiment=embodiment)
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2, placement_layouts_path=str(path))).make_registered()
    try:
        base = env.unwrapped
        assert set(base.scene.articulations) == set(keys)
        validate_recording_assets(env, arena.get_placement_assets())
        scene_cfg = embodiment.get_scene_cfg()
        assert scene_cfg.left_robot.init_state.pos == poses["left_robot"][0].position_xyz
        assert scene_cfg.right_robot.init_state.pos == poses["right_robot"][0].position_xyz
        with pytest.raises(AssertionError, match="Pass scene_assets"):
            validate_recording_assets(env, [])
        env.reset()
        for key in keys:
            expected = torch.stack([pose.to_tensor(base.device) for pose in poses[key][:2]])
            torch.testing.assert_close(base.arena_world.get_pose_e(key), expected, atol=2e-5, rtol=0)

        before = {key: base.arena_world.get_pose_e(key) for key in keys}
        env_ids = torch.tensor([1], device=base.device)
        for key in keys:
            body = base.scene.articulations[key]
            moved = body.data.root_pose_w.torch[env_ids].clone()
            moved[:, 0] += 0.5
            body.write_root_pose_to_sim(moved, env_ids=env_ids)
        base._reset_idx(env_ids)
        for key in keys:
            actual = base.arena_world.get_pose_e(key)
            torch.testing.assert_close(actual[0], before[key][0], atol=2e-5, rtol=0)
            torch.testing.assert_close(actual[1], poses[key][2].to_tensor(base.device), atol=2e-5, rtol=0)
            torch.testing.assert_close(
                base.scene.articulations[key].data.root_vel_w.torch[env_ids],
                torch.zeros((1, 6), device=base.device),
                atol=0,
                rtol=0,
            )
        measured_poses = {}
        for key in keys:
            values = before[key][0].tolist()
            measured_poses[key] = [Pose(tuple(values[:3]), tuple(values[3:]))]
        measured = PlacementLayouts(measured_poses)
        measured.validate_assets(arena.get_placement_assets())
    finally:
        env.close()
    return True


def test_bimanual_root_recording_and_replay(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_bimanual_root_recording_and_replay, tmp_path=tmp_path)


def _test_cached_reference_preserves_joint_reset(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.recording import validate_recording_assets
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_background_physics_reset import _create_background_usds
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app
    from isaaclab_arena.utils.pose import Pose

    background_path = tmp_path / "background.usd"
    online_path = tmp_path / "online_asset.usd"
    _create_background_usds(str(background_path), str(online_path), include_joint_network=False)
    key = "referenced_articulation"
    layouts = None
    for cached in (False, True):
        background = Background("background", str(background_path), object_min_z=0.0)
        reference = ObjectReference(
            name=key,
            prim_path="{ENV_REGEX_NS}/background/referenced_articulation",
            parent_asset=background,
            object_type=ObjectType.ARTICULATION,
        )
        arena = IsaacLabArenaEnvironment(
            name=f"reference_joint_reset_{cached}",
            scene=Scene(assets=[background, reference]),
            placement_layouts=layouts,
        )
        env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2, device="cpu")).make_registered()
        try:
            base = env.unwrapped
            env.reset()
            validate_recording_assets(env, arena.get_placement_assets())
            articulation = base.scene.articulations[key]
            initial_root = articulation.data.root_pose_w.torch.clone()
            initial_position = articulation.data.default_joint_pos.torch.clone()
            initial_velocity = articulation.data.default_joint_vel.torch.clone()
            assert initial_position.shape == (2, 1)
            if cached:
                expected_pose = layouts.poses[key][0].to_tensor(base.device)
                torch.testing.assert_close(base.arena_world.get_pose_e(key)[0], expected_pose, atol=1e-5, rtol=0)
            else:
                local_pose = base.arena_world.get_pose_e(key)[0].tolist()
                # The cached root must win even when it differs from the reference's source pose.
                local_pose[0] += 0.1
                layouts = PlacementLayouts({key: [Pose(tuple(local_pose[:3]), tuple(local_pose[3:]))]})

            moved_root = initial_root.clone()
            moved_root[:, 0] += 0.4
            moved_position = initial_position + 0.2
            moved_velocity = torch.full_like(initial_velocity, 0.5)
            moved_root_velocity = torch.ones_like(articulation.data.root_vel_w.torch)
            articulation.write_root_pose_to_sim(moved_root)
            articulation.write_root_velocity_to_sim(moved_root_velocity)
            articulation.write_joint_position_to_sim_index(position=moved_position)
            articulation.write_joint_velocity_to_sim_index(velocity=moved_velocity)

            # Public partial reset must restore env 1 without touching env 0's current episode.
            base.reset(env_ids=torch.tensor([1], device=base.device))
            torch.testing.assert_close(articulation.data.root_pose_w.torch[1], initial_root[1], atol=1e-5, rtol=0)
            torch.testing.assert_close(articulation.data.root_pose_w.torch[0], moved_root[0], atol=1e-5, rtol=0)
            torch.testing.assert_close(
                articulation.data.root_vel_w.torch[1], torch.zeros_like(moved_root_velocity[1]), atol=0, rtol=0
            )
            torch.testing.assert_close(articulation.data.root_vel_w.torch[0], moved_root_velocity[0], atol=0, rtol=0)
            for state, initial, moved in (
                (articulation.data.joint_pos.torch, initial_position, moved_position),
                (articulation.data.joint_vel.torch, initial_velocity, moved_velocity),
            ):
                torch.testing.assert_close(state[0], moved[0], atol=1e-5, rtol=0)
                torch.testing.assert_close(state[1], initial[1], atol=1e-5, rtol=0, msg=f"cached={cached}: {state}")
        finally:
            env.close()
            if not cached:
                teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
    return True


def test_cached_reference_preserves_joint_reset(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_cached_reference_preserves_joint_reset, tmp_path=tmp_path)
