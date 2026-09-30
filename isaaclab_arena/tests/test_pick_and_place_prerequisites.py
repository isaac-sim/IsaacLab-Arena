# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise settling prerequisites and lift references with live contact physics."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _create_cuboid_usd(path, dimensions, kinematic=False):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Object/geometry")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.2)
    stage.GetRootLayer().Save()


def _object_reached_raised_support(env):
    return env.arena_world.get_pose_e("object")[:, 0] > 0.8


def _build_pick_and_place_env(asset_directory, initial_heights, sequential=False):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import ProceduralTable
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    object_path = f"{asset_directory}/object.usd"
    destination_path = f"{asset_directory}/destination.usd"
    _create_cuboid_usd(object_path, (0.1, 0.1, 0.1))
    _create_cuboid_usd(destination_path, (0.8, 1.5, 0.04), kinematic=True)

    table = ProceduralTable(initial_pose=Pose(position_xyz=(0.0, 0.0, 0.45)))
    raised_support = ProceduralTable(
        instance_name="raised_support",
        prim_path="{ENV_REGEX_NS}/raised_support",
        initial_pose=Pose(position_xyz=(1.0, 0.0, 1.0)),
    )
    destination = Object(
        name="destination",
        usd_path=destination_path,
        object_type=ObjectType.RIGID,
        initial_pose=Pose(position_xyz=(2.0, 0.0, 0.45)),
    )
    picked_object = Object(name="object", usd_path=object_path, object_type=ObjectType.RIGID)
    picked_object.set_initial_pose(
        PosePerEnv(poses=[Pose(position_xyz=(0.0, 0.0, height)) for height in initial_heights])
    )
    task = PickAndPlaceTask(picked_object, destination, table, settling_steps=5)
    if sequential:

        class MoveToRaisedSupportTask(NoTask):
            def get_termination_cfg(self):
                return TaskTerminationCfg(
                    timeout_s=30.0,
                    success=[
                        CompletionCriteria(
                            name="move_to_support",
                            predicate_sequence=[_object_reached_raised_support],
                        )
                    ],
                )

            def get_metrics(self):
                return []

        task = CompositeTaskBase(
            [MoveToRaisedSupportTask(), task],
            episode_length_s=30.0,
            subtasks_are_sequential=True,
        )

    environment = IsaacLabArenaEnvironment(
        name="pick_and_place_prerequisites",
        scene=Scene(assets=[table, raised_support, destination, picked_object]),
        task=task,
    )
    env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    env.reset()
    return env


def _move_asset(env, asset_name, env_ids, positions):
    """Move selected rigid objects to environment-relative positions with zero velocity."""
    import torch

    base_env = env.unwrapped
    env_ids = torch.tensor(env_ids, device=base_env.device)
    T_W_O = base_env.arena_world.get_pose_w(asset_name)[env_ids].clone()
    T_W_O[:, :3] = torch.as_tensor(positions, device=base_env.device) + base_env.scene.env_origins[env_ids]
    asset = base_env.scene[asset_name]
    asset.write_root_pose_to_sim_index(root_pose=T_W_O, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), device=base_env.device),
        env_ids=env_ids,
    )


def _step_without_termination(env, actions):
    _, _, terminated, truncated, info = env.step(actions)
    assert not terminated.any(), "The task terminated before placement."
    assert not truncated.any(), "The task timed out before placement."
    return info["progress_tracking"]


def _test_settling_is_unscored_and_independent_per_environment(_simulation_app):
    import tempfile
    import torch

    with tempfile.TemporaryDirectory() as asset_directory:
        env = _build_pick_and_place_env(asset_directory, initial_heights=(0.52, 2.0))
        try:
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            first_ready_steps = [None, None]
            for step_index in range(1, 81):
                progress = _step_without_termination(env, actions)
                for env_index, state in enumerate(progress["states"]):
                    criteria = state.criteria_by_name["pick_and_place"]
                    assert state.overall_score == criteria.score == 0.0
                    assert progress["events"][env_index] == []
                    if criteria.prerequisites_met and first_ready_steps[env_index] is None:
                        first_ready_steps[env_index] = step_index
                if all(step is not None for step in first_ready_steps):
                    break
            else:
                raise AssertionError("Both objects must settle within 80 control steps.")

            assert 5 <= first_ready_steps[0] < first_ready_steps[1]
            assert env.unwrapped.episode_length_buf.tolist() == [step_index, step_index]
            resting_positions = env.unwrapped.arena_world.get_pose_e("object")[:, :3].clone()
            resting_positions[:, 2] += 0.15
            _move_asset(env, "object", [0, 1], resting_positions)
            progress = _step_without_termination(env, actions)
            for env_index, state in enumerate(progress["states"]):
                assert state.overall_score == state.criteria_by_name["pick_and_place"].score == 0.5
                assert len(progress["events"][env_index]) == 1
                assert progress["events"][env_index][0].score_delta == 0.5

            # Gravity must bring each object into supported contact with the destination.
            _move_asset(env, "object", [0, 1], [(2.0, 0.0, 0.6), (2.0, 0.0, 0.6)])
            completed_envs = set()
            for _ in range(60):
                _, _, terminated, truncated, info = env.step(actions)
                assert not truncated.any()
                for env_index in terminated.nonzero(as_tuple=False).flatten().tolist():
                    state = info["progress_tracking"]["states"][env_index]
                    assert state.overall_score == state.criteria_by_name["pick_and_place"].score == 1.0
                    assert state.all_complete
                    events = info["progress_tracking"]["events"][env_index]
                    assert [event.score_delta for event in events] == [0.5, 0.5]
                    assert events[1].step > events[0].step
                    completed_envs.add(env_index)
                if completed_envs == {0, 1}:
                    break
            assert completed_envs == {0, 1}, "Both supported placements must complete."
        finally:
            env.close()
    return True


def test_settling_is_unscored_and_independent_per_environment():
    assert run_function_with_persistent_simulation_app(_test_settling_is_unscored_and_independent_per_environment)


def _test_delayed_activation_and_partial_reset_use_independent_lift_references(
    _simulation_app,
):
    import tempfile
    import torch

    criteria_name = "subtask_1/pick_and_place"
    with tempfile.TemporaryDirectory() as asset_directory:
        env = _build_pick_and_place_env(asset_directory, initial_heights=(0.52, 0.52), sequential=True)
        try:
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(15):
                progress = _step_without_termination(env, actions)
                for state in progress["states"]:
                    assert state.overall_score == 0.0
                    assert not state.criteria_by_name[criteria_name].prerequisites_met
                assert progress["events"] == [[], []]

            # The second subtask must capture the raised resting pose when it becomes active.
            _move_asset(env, "object", [0, 1], [(1.0, 0.0, 1.07), (1.0, 0.0, 1.07)])
            progress = _step_without_termination(env, actions)
            for state in progress["states"]:
                assert state.criteria_by_name["subtask_0/move_to_support"].is_complete
                assert not state.criteria_by_name[criteria_name].prerequisites_met

            for active_step in range(1, 81):
                progress = _step_without_termination(env, actions)
                ready = []
                for env_index, state in enumerate(progress["states"]):
                    criteria = state.criteria_by_name[criteria_name]
                    assert criteria.score == 0.0
                    assert len(progress["events"][env_index]) == 1
                    if active_step < 5:
                        assert not criteria.prerequisites_met
                    ready.append(criteria.prerequisites_met)
                if all(ready):
                    break
            else:
                raise AssertionError("Both delayed subtasks must become ready.")

            env.unwrapped.reset(env_ids=torch.tensor([0], device=env.unwrapped.device))
            reset_states = env.unwrapped.progress_tracker.get_state()
            assert not reset_states[0].criteria_by_name[criteria_name].prerequisites_met
            assert reset_states[1].criteria_by_name[criteria_name].prerequisites_met
            assert env.unwrapped.progress_tracker.get_events()[0] == []

            # Env 0 starts again at a lower height; env 1 retains its raised reference after moving down.
            _move_asset(env, "raised_support", [0], [(1.0, 0.0, 0.7)])
            _move_asset(env, "object", [0, 1], [(1.0, 0.0, 0.77), (0.0, 0.0, 0.52)])
            progress = _step_without_termination(env, actions)
            assert not progress["states"][0].criteria_by_name[criteria_name].prerequisites_met
            for active_step in range(1, 81):
                progress = _step_without_termination(env, actions)
                reset_criteria = progress["states"][0].criteria_by_name[criteria_name]
                retained_criteria = progress["states"][1].criteria_by_name[criteria_name]
                assert reset_criteria.score == retained_criteria.score == 0.0
                assert retained_criteria.prerequisites_met
                if active_step < 5:
                    assert not reset_criteria.prerequisites_met
                if reset_criteria.prerequisites_met:
                    break
            else:
                raise AssertionError("The reset environment must settle again.")

            _move_asset(env, "object", [0, 1], [(1.0, 0.0, 0.92), (0.0, 0.0, 0.67)])
            progress = _step_without_termination(env, actions)
            assert progress["states"][0].criteria_by_name[criteria_name].score == 0.5
            assert progress["states"][1].criteria_by_name[criteria_name].score == 0.0
            _move_asset(env, "object", [1], [(0.0, 0.0, 1.22)])
            progress = _step_without_termination(env, actions)
            assert progress["states"][1].criteria_by_name[criteria_name].score == 0.5

            retained_events = env.unwrapped.progress_tracker.get_events()[1]
            env.unwrapped.reset(env_ids=torch.tensor([0], device=env.unwrapped.device))
            reset_states = env.unwrapped.progress_tracker.get_state()
            assert reset_states[0].overall_score == 0.0
            assert not reset_states[0].criteria_by_name[criteria_name].prerequisites_met
            assert reset_states[1].criteria_by_name[criteria_name].score == 0.5
            assert reset_states[1].criteria_by_name[criteria_name].prerequisites_met
            assert env.unwrapped.progress_tracker.get_events() == [[], retained_events]
        finally:
            env.close()
    return True


def test_delayed_activation_and_partial_reset_use_independent_lift_references():
    assert run_function_with_persistent_simulation_app(
        _test_delayed_activation_and_partial_reset_use_independent_lift_references
    )
