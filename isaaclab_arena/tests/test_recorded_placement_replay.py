# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end recorded placement replay coverage."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "test_data/placement_replay.yaml"
LAYOUTS = SOURCE.with_suffix(".jsonl")


def _test_recorded_placement_replay_applies_complete_layouts(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_sampler import placement_samples_to_pose_columns, read_placement_samples
    from isaaclab_arena.relations.relation_solver import RelationSolver

    poses_by_root = placement_samples_to_pose_columns(read_placement_samples(LAYOUTS))
    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Recorded replay must not solve")):
        env = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(
                num_envs=3,
                solve_relations=False,
                resolve_on_reset=False,
                recorded_variation_samples_path=str(LAYOUTS),
            ),
        ).make_registered()
    try:
        base = env.unwrapped
        env.reset()
        for name, poses in poses_by_root.items():
            expected = torch.stack([pose.to_tensor(base.device) for pose in poses[:3]])
            torch.testing.assert_close(base.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)

        # The initial reset consumed source rows 0, 1, and 2. A partial reset
        # receives row 3 without disturbing surviving env slots.
        before = {name: base.arena_world.get_pose_e(name).clone() for name in poses_by_root}
        env_ids = torch.tensor([1], device=base.device)
        for name in poses_by_root:
            base.scene[name].write_root_velocity_to_sim(torch.ones((1, 6), device=base.device), env_ids=env_ids)
        base.reset(env_ids=env_ids)
        for name, poses in poses_by_root.items():
            actual = base.arena_world.get_pose_e(name)
            torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
            torch.testing.assert_close(actual[1], poses[3].to_tensor(base.device), atol=2e-5, rtol=0)
            torch.testing.assert_close(
                base.scene[name].data.root_vel_w.torch[env_ids],
                torch.zeros((1, 6), device=base.device),
                atol=0,
                rtol=0,
            )

        assert base.variation_recorder.placement_record is not None
        assert (
            base.variation_recorder.placement_record.sample_for_episode(1, base.get_episode_index(1))["layout_id"]
            == "layout_000003"
        )
    finally:
        env.close()
    return True


def test_recorded_placement_replay_applies_complete_layouts():
    assert run_function_with_persistent_simulation_app(_test_recorded_placement_replay_applies_complete_layouts)


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


def _test_recorded_placement_replays_all_compound_asset_roots(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.placement_sampler import PlacementSample, write_placement_samples
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    poses = {
        "left_robot": [Pose((0.0, 0.5, 0.0)), Pose((0.2, 0.7, 0.0)), Pose((-0.2, 0.6, 0.0))],
        "right_robot": [Pose((0.1, -0.5, 0.0)), Pose((-0.1, -0.7, 0.0)), Pose((0.3, -0.6, 0.0))],
    }
    samples = [
        PlacementSample(
            layout_id=f"layout_{index:06d}",
            poses={name: root_poses[index] for name, root_poses in poses.items()},
        )
        for index in range(3)
    ]
    path = tmp_path / "compound_replay.jsonl"
    write_placement_samples(path, samples)

    embodiment = _make_two_root_embodiment(tmp_path)
    arena = IsaacLabArenaEnvironment(name="compound_root_replay", scene=Scene(assets=[]), embodiment=embodiment)
    env = ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(
            num_envs=2,
            solve_relations=False,
            recorded_variation_samples_path=str(path),
        ),
    ).make_registered()
    try:
        base = env.unwrapped
        env.reset()
        assert set(base.scene.articulations) == set(poses)
        for name, root_poses in poses.items():
            expected = torch.stack([pose.to_tensor(base.device) for pose in root_poses[:2]])
            torch.testing.assert_close(base.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)

        before = {name: base.arena_world.get_pose_e(name).clone() for name in poses}
        env_ids = torch.tensor([1], device=base.device)
        for name in poses:
            body = base.scene.articulations[name]
            moved = body.data.root_pose_w.torch[env_ids].clone()
            moved[:, 0] += 0.5
            body.write_root_pose_to_sim(moved, env_ids=env_ids)
        base.reset(env_ids=env_ids)
        for name, root_poses in poses.items():
            actual = base.arena_world.get_pose_e(name)
            torch.testing.assert_close(actual[0], before[name][0], atol=2e-5, rtol=0)
            torch.testing.assert_close(actual[1], root_poses[2].to_tensor(base.device), atol=2e-5, rtol=0)
    finally:
        env.close()
    return True


def test_recorded_placement_replays_all_compound_asset_roots(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_recorded_placement_replays_all_compound_asset_roots,
        tmp_path=tmp_path,
    )


def _test_recorded_root_replay_preserves_articulation_joint_reset(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.placement_sampler import PlacementSample, write_placement_samples
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_background_physics_reset import _create_background_usds
    from isaaclab_arena.utils.pose import Pose

    background_path = tmp_path / "background.usd"
    online_path = tmp_path / "online_asset.usd"
    _create_background_usds(str(background_path), str(online_path), include_joint_network=False)
    key = "referenced_articulation"
    background = Background("background", str(background_path), object_min_z=0.0)
    reference = ObjectReference(
        name=key,
        prim_path="{ENV_REGEX_NS}/background/referenced_articulation",
        parent_asset=background,
        object_type=ObjectType.ARTICULATION,
    )
    initial_pose = reference.get_initial_pose()
    recorded_pose = Pose(
        (initial_pose.position_xyz[0] + 0.1, *initial_pose.position_xyz[1:]),
        initial_pose.rotation_xyzw,
    )
    path = tmp_path / "articulation_replay.jsonl"
    write_placement_samples(
        path,
        [PlacementSample(layout_id="layout_000000", poses={key: recorded_pose})],
    )
    arena = IsaacLabArenaEnvironment(
        name="reference_joint_replay",
        scene=Scene(assets=[background, reference]),
    )
    env = ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(
            num_envs=2,
            device="cpu",
            solve_relations=False,
            recorded_variation_samples_path=str(path),
        ),
    ).make_registered()
    try:
        base = env.unwrapped
        env.reset()
        articulation = base.scene.articulations[key]
        expected_root = recorded_pose.to_tensor(base.device)
        torch.testing.assert_close(base.arena_world.get_pose_e(key)[0], expected_root, atol=1e-5, rtol=0)
        initial_position = articulation.data.default_joint_pos.torch.clone()
        initial_velocity = articulation.data.default_joint_vel.torch.clone()

        moved_root = articulation.data.root_pose_w.torch.clone()
        moved_root[:, 0] += 0.4
        moved_position = initial_position + 0.2
        moved_velocity = torch.full_like(initial_velocity, 0.5)
        moved_root_velocity = torch.ones_like(articulation.data.root_vel_w.torch)
        articulation.write_root_pose_to_sim(moved_root)
        articulation.write_root_velocity_to_sim(moved_root_velocity)
        articulation.write_joint_position_to_sim_index(position=moved_position)
        articulation.write_joint_velocity_to_sim_index(velocity=moved_velocity)

        env_ids = torch.tensor([1], device=base.device)
        base.reset(env_ids=env_ids)
        torch.testing.assert_close(base.arena_world.get_pose_e(key)[1], expected_root, atol=1e-5, rtol=0)
        torch.testing.assert_close(articulation.data.root_pose_w.torch[0], moved_root[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(
            articulation.data.root_vel_w.torch[1],
            torch.zeros_like(moved_root_velocity[1]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(articulation.data.root_vel_w.torch[0], moved_root_velocity[0], atol=0, rtol=0)
        for state, initial, moved in (
            (articulation.data.joint_pos.torch, initial_position, moved_position),
            (articulation.data.joint_vel.torch, initial_velocity, moved_velocity),
        ):
            torch.testing.assert_close(state[0], moved[0], atol=1e-5, rtol=0)
            torch.testing.assert_close(state[1], initial[1], atol=1e-5, rtol=0)
    finally:
        env.close()
    return True


def test_recorded_root_replay_preserves_articulation_joint_reset(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_recorded_root_replay_preserves_articulation_joint_reset,
        tmp_path=tmp_path,
    )
